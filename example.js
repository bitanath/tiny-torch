import {nn,torch} from "./dist/index.mjs"
import * as weights from "./weights.json" with {type: "json"}

const vocab_size = 27
const embedding_dim = 256
const hidden_layers = 320
const model = new torch.models.SimpleRNN(vocab_size, embedding_dim, hidden_layers)
const [charToIndex,indexToChar] = character_converts()
let initialized = false

function character_converts(){
    const text = "abcdefghijklmnopqrstuvwxyz";
    const chars = Array.from(new Set(text)).sort();
    chars.unshift('\0');

    const charToIndex = {};
    chars.forEach((char, index) => {
        charToIndex[char] = index;
    })

    const indexToChar = {};
    chars.forEach((char, index) => {
        indexToChar[index] = char;
    })

    return [charToIndex,indexToChar]
}

function inference(input_string){
    if(!initialized){
        console.log("INIT WITH WEIGHTS")
        model.embed.E = torch.tensor(weights.default["embedding.weight"])
        model.rnn.W_hh = torch.tensor(weights.default["rnn.W_hh"])
        model.rnn.W_ih = torch.tensor(weights.default["rnn.W_ih"])
        model.rnn.b_hh = torch.tensor(weights.default["rnn.b_hh"])
        model.rnn.b_ih = torch.tensor(weights.default["rnn.b_ih"])
        model.dense.W = torch.transpose(torch.tensor(weights.default["dense.weight"]),0,1) //NOTE: linear weights stored (vocab_size,hidden_layers) in Pytorch
        model.dense.b = torch.tensor(weights.default["dense.bias"])
        initialized = true
    }

    const arr = input_string.split('').map(char => charToIndex[char]);
    const tensor = torch.tensor(arr,false,"cpu").unsqueeze(0) //requires_grad is False, device is CPU

    const prediction = model.forward(tensor)
    const argmaxed = torch.argmax(prediction).data
    console.log("Argmaxed",argmaxed[0],indexToChar[argmaxed[0]],input_string+indexToChar[argmaxed[0]])
}

inference('sta')
inference('thematr')
inference('thegodfathe')
inference('aquama')