import { PositionalEmbedding,Embedding,Block,Module,LayerNorm,Linear } from "./layers.js";
import { add,Tensor,Parameter,zeros,transpose } from "./tensor.js";
import * as init from "./init.js"


class RecurrentLayer extends Module{
  public W_ih: Parameter; // Input-to-hidden weights
  public W_hh: Parameter; // Hidden-to-hidden weights
  public b_ih: Parameter; // Input-to-hidden bias
  public b_hh: Parameter; // Hidden-to-hidden bias

  constructor(input_size:number, hidden_size:number, nonlinearity='tanh'){
    super()
    this.input_size = input_size
    this.hidden_size = hidden_size
    this.nonlinearity = nonlinearity

    this.W_ih = new Parameter(init.zeros_([hidden_size, input_size]))
    this.W_hh = new Parameter(init.zeros_([hidden_size, hidden_size]))
    this.b_ih = new Parameter(init.zeros_([hidden_size]))
    this.b_hh = new Parameter(init.zeros_([hidden_size]))

    // this.reset_parameters();
  }

  reset_parameters(): void {
    this.W_ih = init.kaiming_uniform_(this.W_ih, Math.sqrt(5)) as Parameter;
    this.W_hh = init.kaiming_uniform_(this.W_hh, Math.sqrt(5)) as Parameter;
    
    const { fanIn } = init._calculate_fan_in_and_fan_out([this.hidden_size, this.input_size]);
    const bound = fanIn > 0 ? 1 / Math.sqrt(fanIn) : 0;
    
    this.b_ih = init.uniform_(this.b_ih, -bound, bound) as Parameter;
    this.b_hh = init.uniform_(this.b_hh, -bound, bound) as Parameter;
  }

  linear(x_t: Tensor, weight: Parameter, bias: Parameter): Tensor {
    const output = x_t.matmul(transpose(weight, 0, 1)); //x @ weight.T in PyTorch
    if (bias !== null) {
      return add(output, bias);
    }
    return output;
  }

  /**
   * Forward pass of the recurrent layer
   * @param x - Input tensor of shape [batch_size, seq_len, input_size]
   * @param h_0 - Optional initial hidden state of shape [batch_size, hidden_size]
   * @returns Tuple of [output, h_n] where:
   *          - output is of shape [batch_size, seq_len, hidden_size]
   *          - h_n is the final hidden state of shape [batch_size, hidden_size]
   */
  forward(x: Tensor, h_0?: Tensor): [Tensor, Tensor] {
    const shape = x.shape;
    const batch_size = shape[0];
    const seq_len = shape[1];
    
    let h_t: Tensor;
    if (!h_0) {
      h_t = zeros([batch_size, this.hidden_size],x.requires_grad,x.device)
    } else {
      h_t = h_0;
    }

    const output = zeros([batch_size, seq_len, this.hidden_size],x.requires_grad,x.device)
    
    for (let t = 0; t < seq_len; t++) {
      const x_t = x.slice([null, t, null]); // Equivalent to x[:, t, :]
      
      const input_projection = this.linear(x_t, this.W_ih, this.b_ih);
      const hidden_projection = this.linear(h_t, this.W_hh, this.b_hh);
      const combined = input_projection.add(hidden_projection);
      
      h_t = init.tanh(combined);
      output.setSlice([null, t, null], h_t);
    }
    
    return [output, h_t];
  }

}

export class SimpleRNN extends Module{
  public embed: Embedding
  public rnn:RecurrentLayer
  public dense:Linear
  constructor(vocab_size:number=27, embedding_dim:number=256, hidden_layers=320){
    super()
    this.embed = new Embedding(vocab_size, embedding_dim)
    this.rnn = new RecurrentLayer(embedding_dim, hidden_layers)  
    this.dense = new Linear(hidden_layers, vocab_size)

    this.dense.W = init.xavier_normal_(this.dense.W)
    this.embed.E = init.xavier_normal_(this.embed.E)
  }

  forward(x:Tensor) {
      const embedded = this.embed.forward(x)
      let [out,_] = this.rnn.forward(embedded)
      out = this.dense.forward(out.slice([null, out.shape[1] - 1, null])) 
      return out
  }
}

export class Transformer extends Module {
  constructor(vocab_size:number=27, hidden_size:number=64, n_timesteps:number=16, n_heads:number=4, dropout_p:number=0.2, device:string='cpu') {
    super();
    // Instantiate Transformer's Layers:
    this.embed = new Embedding(vocab_size, hidden_size);
    this.pos_embed = new PositionalEmbedding(n_timesteps, hidden_size);
    this.b1 = new Block(hidden_size, hidden_size, n_heads, n_timesteps, dropout_p, device);
    this.b2 = new Block(hidden_size, hidden_size, n_heads, n_timesteps, dropout_p, device);
    this.ln = new LayerNorm(hidden_size);
    this.linear = new Linear(hidden_size, vocab_size, device);
  }

  forward(x:Tensor) {
    let z;
    z = add(this.embed.forward(x), this.pos_embed.forward(x));
    z = this.b1.forward(z);
    z = this.b2.forward(z);
    z = this.ln.forward(z);
    z = this.linear.forward(z);
    return z;
  }
}