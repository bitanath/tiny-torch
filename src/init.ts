import { Tensor } from "./tensor.js";

export function tanh(x: Tensor): Tensor {
    function _tanh(x: Array<any>): Array<any> {
      if (typeof x[0] === "number") {
        return x.map((el: number): number => {
          // tanh(x) = (e^x - e^-x) / (e^x + e^-x)
          const exp_x = Math.exp(el);
          const exp_neg_x = Math.exp(-el);
          return (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
        });
      } else if (typeof x[0] === "object") {
        return x.map((el: Array<any>): Array<any> => _tanh(el));
      } else {
        throw Error("In tanh, provided Tensor is not homogenous.");
      }
    }
    const tensor = new Tensor(_tanh(x._data),x.requires_grad,x.device)
    return tensor;
}

export function _calculate_fan_in_and_fan_out(shape: number[]): { fanIn: number, fanOut: number } {
    if (shape.length < 2) {
      throw new Error('Tensor must have at least 2 dimensions for fan in/out calculation');
    }
    
    let fanIn: number;
    let fanOut: number;
    
    if (shape.length === 2) {
      fanIn = shape[1];
      fanOut = shape[0];
    } else {
      const receptiveFieldSize = shape.slice(2).reduce((a, b) => a * b, 1);
      fanIn = shape[1] * receptiveFieldSize;
      fanOut = shape[0] * receptiveFieldSize;
    }
    
    return { fanIn, fanOut };
}

export function xavier_normal_(tensor: Tensor, gain: number = 1.0): Tensor {
    const randomNormal = (mean: number = 0, std: number = 1): number => {
      // Box-Muller transform
      const u1 = Math.random();
      const u2 = Math.random();
      
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      return mean + z0 * std;
    };
    
    const fillTensorData = (data: Array<any> | number, std: number): Array<any> | number => {
      if (Array.isArray(data)) {
        return data.map(item => fillTensorData(item, std));
      } else {
        return randomNormal(0, std);
      }
    };
    
    const { fanIn, fanOut } = _calculate_fan_in_and_fan_out(tensor.shape);
    const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
    
    const newData = fillTensorData(tensor.data, std);
    const newTensor = new Tensor(newData,tensor.requires_grad,tensor.device)
    
    return newTensor;
}
  
export function uniform_(tensor: Tensor, a: number = 0.0, b: number = 1.0): Tensor {
    const randomUniform = (): number => {
      return a + Math.random() * (b - a);
    };
    
    const createNewData = (data: Array<any> | number): Array<any> | number => {
      if (Array.isArray(data)) {
        return data.map(item => createNewData(item));
      } else {
        return randomUniform();
      }
    };
    
    const newData = createNewData(tensor.data);
    const newTensor = new Tensor(newData,tensor.requires_grad,tensor.device)
    
    return newTensor
}

export function kaiming_uniform_( tensor: Tensor, a: number = Math.sqrt(5), mode: 'fan_in' | 'fan_out' = 'fan_in', nonlinearity: string = 'leaky_relu' ): Tensor {
    let gain: number;
    if (nonlinearity === 'leaky_relu') {
      gain = Math.sqrt(2.0 / (1 + a * a));
    } else if (nonlinearity === 'relu') {
      gain = Math.sqrt(2.0);
    } else if (nonlinearity === 'tanh') {
      gain = 5.0 / 3.0;
    } else {
      gain = 1.0;
    }
    
    const { fanIn, fanOut } = _calculate_fan_in_and_fan_out(tensor.shape);
    const fan = mode === 'fan_in' ? fanIn : fanOut;
    const bound = gain * Math.sqrt(3.0 / fan);
    
    return uniform_(tensor, -bound, bound);
}

export function zeros_(shape: number[]): any[] {
    if (shape.length === 1) {
      return Array(shape[0]).fill(0);
    }
    
    return Array(shape[0]).fill(0).map(() => 
        zeros_(shape.slice(1))
    );
}