import torch
import torch.nn as nn
import numpy as np
from utils import load_coco_data,sample_coco_minibatch,decode_captions
from Transformer_layer import *
from Img_utils import *
import matplotlib.pyplot as plt
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider


class CaptioningSolverTransformer(object):
    def __init__(self, model, data, idx_to_word, **kwargs):
        self.model = model
        self.data = data

        # Unpack keyword arguments
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)

        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        self._reset()

        self.idx_to_word = idx_to_word

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.loss_history = []


    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        minibatch = sample_coco_minibatch(
            self.data, batch_size=self.batch_size, split="train"
        )
        captions, features, urls = minibatch

        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = captions_out != self.model._null

        t_features = torch.Tensor(features)
        t_captions_in = torch.LongTensor(captions_in)
        t_captions_out = torch.LongTensor(captions_out)
        t_mask = torch.LongTensor(mask)
        logits = self.model(t_features, t_captions_in)

        loss = self.transformer_temporal_softmax_loss(logits, t_captions_out, t_mask)
        self.loss_history.append(loss.detach().numpy())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.data["train_captions"].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )
            epoch_end = (t + 1) % iterations_per_epoch == 0

    def transformer_temporal_softmax_loss(self, x, y, mask):
        N, T, V = x.shape

        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        loss = torch.nn.functional.cross_entropy(x_flat,  y_flat, reduction='none')
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)

        return loss
    
class CaptioningTransformer(nn.Module):
    def __init__(self, word_to_idx, input_dim, wordvec_dim, num_heads=4,
                 num_layers=2, max_length=50):
        super().__init__()
        vocab_size = len(word_to_idx)
        self.vocab_size = vocab_size
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        self.visual_projection = nn.Linear(input_dim, wordvec_dim)
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(wordvec_dim, max_len=max_length)

        decoder_layer = TransformerDecoderLayer(input_dim=wordvec_dim, num_heads=num_heads)
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)

        self.output = nn.Linear(wordvec_dim, vocab_size)


    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def forward(self, features, captions):
        N,T = captions.shape
        scores = torch.empty(N,T,self.vocab_size)
        # (N, T) -> (N, T, W)
        embedding_caption = self.embedding(captions)
        caption_embeddings = self.positional_encoding(embedding_caption)
        # (N, D) -> (N, W)
        memory = self.visual_projection(features)
        memory = memory.unsqueeze(1)
        # (1 ,T, T)
        tgt_mask = torch.tril(torch.ones((T, T), device=captions.device)).bool()
        # (N, T, W)
        decoder_output = self.transformer(
        tgt=caption_embeddings, 
        memory=memory, 
        tgt_mask=tgt_mask
        )
        # (N, T, V)
        scores = self.output(decoder_output) 
        return scores
    def sample(self, features, max_length=30):
        with torch.no_grad():
            features = torch.Tensor(features)
            N = features.shape[0]
            captions = self._null * np.ones((N, max_length), dtype=np.int32)
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                captions[:, t] = word.numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions
        
def evaluate_captions(data, transformer, split='val', batch_size=50):
    minibatch = sample_coco_minibatch(data, split=split, batch_size=batch_size)
    gt_captions, features, urls = minibatch
    
    gt_decoded = decode_captions(gt_captions, data['idx_to_word'])
    pred = transformer.sample(features, max_length=30)
    pred_decoded = decode_captions(pred, data['idx_to_word'])

    gts = {}
    res = {}
    for i in range(len(pred_decoded)):
        gts[i] = [gt_decoded[i]]   
        res[i] = [pred_decoded[i]] 

    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)
    bleu4 = bleu_scores[3]

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)

    return bleu4, cider_score
        
torch.manual_seed(231)
np.random.seed(231)

data = load_coco_data(max_train=50)

transformer = CaptioningTransformer(
          word_to_idx=data['word_to_idx'],
          input_dim=data['train_features'].shape[1],
          wordvec_dim=256,
          num_heads=2,
          num_layers=2,
          max_length=30
        )

transformer_solver = CaptioningSolverTransformer(transformer, data, idx_to_word=data['idx_to_word'],
           num_epochs=100,
           batch_size=25,
           learning_rate=0.001,
           verbose=True, print_every=10,
         )

transformer_solver.train()

# Plot the training losses.
plt.plot(transformer_solver.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()
for split in ['train', 'val']:
    minibatch = sample_coco_minibatch(data, split=split, batch_size=2)
    gt_captions, features, urls = minibatch
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    sample_captions = transformer.sample(features, max_length=30)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])
    bleu4, cider = evaluate_captions(data, transformer, split=split, batch_size=50)
    print(f"{split.upper()} — BLEU-4: {bleu4:.4f}, CIDEr: {cider:.4f}")

