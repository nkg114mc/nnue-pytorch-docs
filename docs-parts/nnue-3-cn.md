
## Forward pass implementation

In this part we will look at model inference as it could be implemented in a simple chess engine. We will work with floating point values for simplicity here. Input generation is outside of the scope of this implementation.

### Example network

We will take a more generally defined network, with architecture `FeatureSet[N]->M*2->K->1`. The layers will therefore be:

1. `L_0`: Linear `N->M`
2. `C_0`: Clipped ReLu of size `M*2`
3. `L_1`: Linear `M*2->K`
4. `C_1`: Clipped ReLu of size `K`
5. `L_2`: Linear `K->1`

### Layer parameters

Linear layers have 2 parameters - weights and biases. We will refer to them as `L_0.weight` and `L_0.bias` respectively. The layers also contain the number of inputs and outputs, in `L_0.num_inputs` and `L_0.num_outputs` respectively.

Here something important has to be said about layout of the weight matrix. For the sparse multiplication, the column-major (a column is contiguous in memory) layout is favorable, as we're adding columns, but for dense multiplication this is not so clear and a row-major layout may be preferable. For now we will stick to the column-major layout, but we may revisit the row-major one when it comes to quantization and optimization. For now we assume `L_0.weight` allows access to the individual elements in the following form: `L_0.weight[column_index][row_index]`.

The code presented is very close to C++ but technicalities might be ommited.

### Accumulator

The accumulator can be represented by an array that is stored along other position state information on the search stack.

```cpp
struct NnueAccumulator {
    // Two vectors of size N. v[0] for white's, and v[1] for black's perspectives.
    float v[2][N];

    // This will be utilised in later code snippets to make the access less verbose
    float* operator[](Color perspective) {
        return v[perspective];
    }
};
```

The accumulator can either be updated lazily on evaluation, or on each move. It doesn't matter here, but it has to be updated *somehow*. Whether it's better to update lazily or eagerly depends on the number of evaluations done during search. For updates there are two cases, as laid out before:

1. The accumulator has to be recomputed from scratch.
2. The previous accumulator is reused and just updated with changed features

#### Refreshing the accumulator

```cpp
void refresh_accumulator(
    const LinearLayer&      layer,            // this will always be L_0
    NnueAccumulator&        new_acc,          // storage for the result
    const std::vector<int>& active_features,  // the indices of features that are active for this position
    Color                   perspective       // the perspective to refresh
) {
    // First we copy the layer bias, that's our starting point
    for (int i = 0; i < M; ++i) {
        new_acc[perspective][i] = layer.bias[i];
    }

    // Then we just accumulate all the columns for the active features. That's what accumulators do!
    for (int a : active_features) {
        for (int i = 0; i < M; ++i) {
            new_acc[perspective][i] += layer.weight[a][i];
        }
    }
}
```

#### Updating the accumulator

```cpp
void update_accumulator(
    const LinearLayer&      layer,            // this will always be L_0
    NnueAccumulator&        new_acc,          // it's nice to have already provided storage for
                                              // the new accumulator. Relevant parts will be overwritten
    const NNueAccumulator&  prev_acc,         // the previous accumulator, the one we're reusing
    const std::vector<int>& removed_features, // the indices of features that were removed
    const std::vector<int>& added_features,   // the indices of features that were added
    Color                   perspective       // the perspective to update, remember we have two,
                                              // they have separate feature lists, and it even may happen
                                              // that one is updated while the other needs a full refresh
) {
    // First we copy the previous values, that's our starting point
    for (int i = 0; i < M; ++i) {
        new_acc[perspective][i] = prev_acc[perspective][i];
    }

    // Then we subtract the weights of the removed features
    for (int r : removed_features) {
        for (int i = 0; i < M; ++i) {
            // Just subtract r-th column
            new_acc[perspective][i] -= layer.weight[r][i];
        }
    }

    // Similar for the added features, but add instead of subtracting
    for (int a : added_features) {
        for (int i = 0; i < M; ++i) {
            new_acc[perspective][i] += layer.weight[a][i];
        }
    }
}
```

And that's it! Pretty simple, isn't it?

### Linear layer

This is simple vector-matrix multiplication, what could be complicated about it you ask? Nothing for now, but it will get complicated once optimization starts. Right now we won't optimize, but we will at least write a version that uses the fact that the weight matrix has column-major layout.

```cpp
float* linear(
    const LinearLayer& layer,  // the layer to use. We have two: L_1, L_2
    float*             output, // the already allocated storage for the result
    const float*       input   // the input, which is the output of the previous CReLu layer
) {
    // First copy the biases to the output. We will be adding columns on top of it.
    for (int i = 0; i < layer.num_outputs; ++i) {
        output[i] = layer.bias[i];
    }

    // Remember that rainbowy diagram long time ago? This is it.
    // We're adding columns one by one, scaled by the input values.
    for (int i = 0; i < layer.num_inputs; ++i) {
        for (int j = 0; j < layer.num_outputs; ++j) {
            output[j] += input[i] * layer.weight[i][j];
        }
    }

    // Let the caller know where the used buffer ends.
    return output + layer.num_outputs;
}
```

### ClippedReLu

```cpp
float* crelu(,
    int          size,   // no need to have any layer structure, we just need the number of elements
    float*       output, // the already allocated storage for the result
    const float* input   // the input, which is the output of the previous linear layer
) {
    for (int i = 0; i < size; ++i) {
        output[i] = min(max(input[i], 0), 1);
    }

    return output + size;
}
```

### Putting it together

In a crude pseudo code. The feature index generation is left as an exercise for the reader.

```cpp
void Position::do_move(...) {
    ... // do the movey stuff

    for (Color perspective : { WHITE, BLACK }) {
        if (needs_refresh[perspective]) {
            refresh_accumulator(
                L_0,
                this->accumulator,
                this->get_active_features(perspective),
                perspective
            );
        } else {
            update_accumulator(
                L_0,
                this->accumulator,
                this->get_previous_position()->accumulator,
                this->get_removed_features(perspective),
                this->get_added_features(perspective),
                perspective
            );
        }
    }
}

float nnue_evaluate(const Position& pos) {
    float buffer[...]; // allocate enough space for the results

    // We need to prepare the input first! We will put the accumulator for
    // the side to move first, and the other second.
    float input[2*M];
    Color stm = pos.side_to_move;
    for (int i = 0; i < M; ++i) {
        input[  i] = pos.accumulator[ stm][i];
        input[M+i] = pos.accumulator[!stm][i];
    }

    float* curr_output = buffer;
    float* curr_input = input;
    float* next_output;

    // Evaluate one layer and move both input and output forward.
    // Last output becomes the next input.
    next_output = crelu(L_0.num_outputs, curr_output, input);
    curr_input = curr_output;
    curr_output = next_output;

    next_output = linear(L_1, curr_output, curr_input);
    curr_input = curr_output;
    curr_output = next_output;

    next_output = crelu(L_1.num_outputs, curr_output, input);
    curr_input = curr_output;
    curr_output = next_output;

    next_output = linear(L_2, curr_output, curr_input);

    // We're done. The last layer should have put 1 value out under *curr_output.
    return *curr_output;
}
```

And that's it! That's the whole network. What do you mean you can't use it?! OH RIGHT, you don't have a net trained, what a bummer.
