
## Architectures and new directions

### Simple HalfKP Stockfish architecture

This is the first architecture used in Stockfish. The only thing that is different is that in this document, we use HalfKP that doesn't have these 64 additional unused features that are a leftover from Shogi. Other than that, there's nothing new, just using the primitives described earlier to put things into perspective.

![](img/HalfKP-40960-256x2-32-32-1.svg)

### HalfKAv2 feature set.

HalfKA feature set was briefly mentioned in this document as a brother of HalfKP. It initially had a small drawback that wasted some space. HalfKAv2 is the improved version that uses 8% less space, but otherwise is identical. What's the difference? Let's consider a subset of features for a given our king square `S`. Normally in HalfKA there are 768 possible features, that is `64*12`, as there is 64 squares and 12 pieces (type + color). But we can notice that with the our king square fixed at `S` we know that the opponent's king is not at `S` - our king uses just 1 feature from the 64 given for it, and the other king only uses 63 (minus our king ring, but it doesn't matter) from its 64 given features, and the two sets are disjoint! So we can merge the two pieces "into one", and reduce the number of buckets from 12 into 11, reducing the size by about 8%. However, care must be taken when applying factorization, as this compression needs to be reverted and a whole `A` subset with 768 features must be used. Otherwise it's possible to mix up king positions, as while the compression is valid for a single `64*11` bucket, it doesn't hold when we try to mix the buckets, as it happens when we factorize the features.

### HalfKAv2_hm feature set.

"hm" here stands for "horizontally mirrored". This feature set is basically HalfKAv2, but the board is assumed to have horizontal symmetry. While this assumption obviously doesn't hold in chess it works well in practice. The idea behind this feature set is to, for each perspective, transform the board such that our king is on the e..h files (by convention, could also be a..d file). Knowing that only half of the king squares will be used allows us to cut the number of input features in half, effectively halving the size of the feature transformer. Stockfish uses this feature set since early august 2021 to support large feature transformers without the need for inconveniently large nets. This has also been used in Scorpio, along other ways of minimizing the network size.

Let's consider an example where the white king is on a1 and the black king on g1. The board will have to be mirrored horizontally for the white's perspetctive to put the king within the e..h files; for black's perspective the king is already within the e..h files. On first sight it may appear that this creates discrepancies between the perspective in cases where the board is mirrored only for one perspective, but it works out remarkably well in practice - the strength difference is hardly measurable.

### A part of the feature transformer directly forwarded to the output.

Normally the nets have hard time learning high material imbalance, or even representing high evaluations at all. But we can help it with that. We already accumulate some 256 values for each piece on the board, does this ring a bell? What if we added one more and designated it to mean "PSQT"? That's what we will do. We will simply make the feature transformer weight row have 257 values, and use the last one as "PSQT". We can help it during training by initializing it to something that resembles good PSQT values (but remember to scale it according to quantization!). But we have two perspectives? What about that? Right, we do, but we can average them, like `(our - their) / 2` (keeping in mind that their must be negated). Handling it in the trainer is quite easy.

```python
wp = self.ft(w_in)
bp = self.ft(b_in)
w, wpsqt = torch.split(wp, wp.shape[1]-1, dim=1)
b, bpsqt = torch.split(bp, bp.shape[1]-1, dim=1)
[...]
y = self.output(l2_) + (wpsqt - bpsqt) * (us - 0.5)
```

We should also use a feature set that includes king features, as it provides additional PSQT values that may be important. So we will use HalfKAv2.

![](img/HalfKAv2-45056-256x2P1x2-32-32-1.svg)

### Multiple PSQT outputs and multiple subnetworks

Until now all networks have been using one PSQT output and one layer stack (that -32-32-1 part in the Stockfish's network; whatever comes after the feature transformer). But what if we could use more? We need to find some easy-to-compute discriminator to choose the outputs/layer stacks by. One such good discriminator is the piece count, as it's cheap to compute, fairly well behaved during the game, and the number of pieces can dramatically change how we look at the position. So let's try 8 buckets for both, based on `(piece_count - 1) / 4`.

![](img/HalfKAv2-45056-256x2P8x2[-32-32-1]x8.svg)

But how to implement it in the trainer? "Choosing stuff" is not very GPU friendly, and we're doing batching too, right? It's not indeed, but thankfully the layers are very small, so we can just evaluate all of them and only choose the results! Moreover, multiple `N` linear layers can just be emulated by a single one with `N` times as many outputs. Let's see how it could be implemented in PyTorch:

```python
# Numbers of hidden neurons
L1 = 256
L2 = 32
L3 = 32

class LayerStacks(nn.Module):
    def __init__(self, count):
        super(LayerStacks, self).__init__()

        self.count = count
        # Layers are larger, very good for GPUs
        self.l1 = nn.Linear(2 * L1, L2 * count)
        self.l2 = nn.Linear(L2, L3 * count)
        self.output = nn.Linear(L3, 1 * count)

        # For caching some magic later.
        self.idx_offset = None

        # Don't forget to initialize the layers to your liking.
        # It might be worth it to initialize the layers in each layer
        # stack identically, or introduce a factorizer for the first
        # layer in the layer stacks.

    def forward(self, x, layer_stack_indices):
        # Precompute and cache the offset for gathers
        if self.idx_offset == None or self.idx_offset.shape[0] != x.shape[0]:
            # This is the "magic". There's no easy way to gather just one thing out of
            # many for each position in the batch, but we can interpret the whole batch as
            # N * batch_size outputs and modify the layer_stack_indices to point to
            # `N * i + layer_stack_indices`, where `i` is the position index.
            # Here we precompute the additive part. This part includes just the values `N * i`
            self.idx_offset = torch.arange(0, x.shape[0] * self.count, self.count, device=layer_stack_indices.device)

        # And here we add the current indices to the additive part.
        indices = layer_stack_indices.flatten() + self.idx_offset

        # Evaluate the whole layer
        l1s_ = self.l1(x)
        # View the output as a `N * batch_size` chunks
        # Choose `batch_size` chunks based on the indices we computed before.
        l1c_ = l1s_.view(-1, L2)[indices]
        # We could have applied ClippedReLU earlier, doesn't matter.
        l1y_ = torch.clamp(l1c_, 0.0, 1.0)

        # Same for the second layer.
        l2s_ = self.l2(l1y_)
        l2c_ = l2s_.view(-1, L3)[indices]
        l2y_ = torch.clamp(l2c_, 0.0, 1.0)

        # Same for the third layer, but no clamping since it's the output.
        l3s_ = self.output(l2y_)
        l3y_ = l3s_.view(-1, 1)[indices]

        return l3y_
```

Handling of the PSQT outputs is easier since the is in fact, a simple way of gathering individual values (we couldn't use it above because we were gathering whole rows):

```python
wp = self.input(w_in)
bp = self.input(b_in)
w, wpsqt = torch.split(wp, wp.shape[1]-8, dim=1)
b, bpsqt = torch.split(bp, bp.shape[1]-8, dim=1)
[...]
psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
wpsqt = wpsqt.gather(1, psqt_indices_unsq)
bpsqt = bpsqt.gather(1, psqt_indices_unsq)
y = self.layer_stacks(l0_, layer_stack_indices) + (wpsqt - bpsqt) * (us - 0.5)
```
