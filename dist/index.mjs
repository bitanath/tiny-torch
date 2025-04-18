import { createRequire } from 'module';
import fs from 'fs';

var require$1 = (
			true
				? /* @__PURE__ */ createRequire(import.meta.url)
				: require
		);

function getShape(data, shape = []) {
  if (data instanceof Array && data.length === 0) {
    return [0];
  }
  if (typeof data === "number") {
    if (JSON.stringify(shape) === "[]") {
      return [1];
    }
    return shape;
  }
  if (typeof data[0] === "number" && Array.isArray(data)) {
    for (const element of data) {
      if (typeof element != "number") {
        throw new Error("The requested array has an inhomogeneous shape.");
      }
    }
    shape.push(data.length);
    return shape;
  }
  if (Array.isArray(data[0])) {
    let elementLength = data[0].length;
    for (const element of data) {
      if (typeof element != "object" && typeof element != "number") {
        throw new Error("TypeError: the input data is not a number.");
      } else if (Array.isArray(element) && elementLength != element.length) {
        throw new Error("The requested array has an inhomogeneous shape.");
      } else if (Array.isArray(element)) {
        elementLength = element.length;
      }
    }
    shape.push(data.length);
  }
  return getShape(data[0], shape);
}
function assureArray(a) {
  if (Array.isArray(a)) {
    return a;
  } else if (typeof a === "number") {
    return [a];
  } else if (a === null) {
    return a;
  }
  return a._data;
}
function getData(a) {
  if (Array.isArray(a)) {
    return a;
  }
  if (typeof a === "number") {
    return a;
  }
  return a._data;
}

class Tensor {
  requires_grad = false;
  _data;
  shape;
  _grad;
  children;
  parents;
  operation;
  visited = false;
  m;
  v;
  device;
  forwardKernel;
  backwardKernelA;
  backwardKernelB;
  batch_size;
  gpu;
  warned;
  /**
   * Creates new instance of the Tensor class.
   * @param {object} data - Iterable containing the data to be stored in the Tensor.
   * @param {boolean} requires_grad - Whether to keep track of this tensor's gradients.
   * @param {string} device - Device to store Tensor. Either "gpu" or "cpu".
   */
  constructor(data, requires_grad = false, device = "cpu") {
    if (typeof data === "object") {
      this._data = data;
    } else if (typeof data === "number") {
      this._data = [data];
    } else {
      throw Error('Your argument "data" is not a number or an iterable.');
    }
    this.shape = getShape(data);
    this.device = device;
    this.requires_grad = requires_grad;
    this.forwardKernel = null;
    this.batch_size = null;
    this.gpu = null;
    this.warned = false;
    if (this.requires_grad) {
      this._grad = zeros(this.shape);
    }
    this.children = [];
    this.parents = [];
    this.operation = null;
    this.visited = false;
  }
  /**
   * Returns the data in the Tensor.
   */
  get data() {
    return this._data;
  }
  /**
   * Returns the data's length'.
   */
  get length() {
    return this._data.length;
  }
  /**
   * Returns the number of dimensions in the Tensor.
   */
  get ndims() {
    return this.shape.length;
  }
  /**
   * Returns the tensor's gradients.
   */
  get grad() {
    return this._grad?.data;
  }
  /**
   * Performs backward pass from THIS tensor backwards.
   * It fills every tensor that originated this one and that has requires_grad=true's gradients to their gradients relative to THIS tensor.
   */
  backward(grad = null, child = null) {
    if (!this.requires_grad) {
      throw new Error("this tensor has requires_grad set to False");
    }
    if (grad === null) {
      grad = ones(this.shape);
      this.children = [];
    }
    this._grad = new Tensor(_add(this._grad?.data, grad.data));
    if (child != null) {
      const idx = this.children.indexOf(child);
      this.children.splice(idx, 1);
    }
    if (this.operation != null) {
      if (this.children.length === 0) {
        this.operation.backward(this._grad, this);
      }
    }
  }
  /**
   * Sends this Tensor to the provided device.
   * @param {string} device - Device to store Tensor. Either "gpu" or "cpu".
   * @param {boolean} requires_grad - Whether to keep track of this tensor's gradients.
   * @param {string} device - gpu or cpu: device to store Tensor.
   */
  to(device) {
    this.device = device;
  }
  /**
   * Reset this Tensor's gradients to zero.
   */
  zero_grad() {
    this._grad = zeros(this.shape);
    this.children = [];
    this.parents = [];
    this.operation = null;
    if (this.m instanceof Tensor && this.v instanceof Tensor) {
      this.m.zero_grad_graph();
      this.v.zero_grad_graph();
    }
  }
  /**
   * Reset the gradients of this Tensor, and of all of the Tensors that led to it.
   */
  zero_grad_graph() {
    this.zero_grad();
    if (this.operation != null) {
      for (const parent of this.parents) {
        parent.zero_grad_graph();
        parent.parents = [];
      }
      this.operation = null;
      this.parents = [];
      this.children = [];
    }
  }
  /**
   * Turns the data in the Tensor into a javascript list object.
   */
  tolist() {
    return this._data;
  }
  /**
   * Returns a single element tensor into a js number
   */
  item() {
    if (this._data.length !== 1) {
      throw new Error("item() can only be called on tensors with a single element");
    }
    return this._data[0];
  }
  /**
   * Removes dimensions of size 1 from the tensor.
   * 
   * @param dim Optional dimension to squeeze. If specified, only squeezes the dimension if it's 1.
   * If not specified, squeezes all dimensions of size 1.
   * @returns A new tensor with selected dimensions of size 1 removed
   */
  squeeze(dim) {
    const newShape = [...this.shape];
    if (dim !== void 0) {
      const actualDim = dim < 0 ? this.shape.length + dim : dim;
      if (actualDim < 0 || actualDim >= this.shape.length) {
        throw new Error(`Dimension out of range (expected to be in range of [${-this.shape.length}, ${this.shape.length - 1}], but got ${dim})`);
      }
      if (this.shape[actualDim] === 1) {
        newShape.splice(actualDim, 1);
      }
    } else {
      for (let i = newShape.length - 1; i >= 0; i--) {
        if (newShape[i] === 1) {
          newShape.splice(i, 1);
        }
      }
    }
    const newTensor = new Tensor([...this._data], this.requires_grad, this.device);
    newTensor.shape = newShape;
    return newTensor;
  }
  /**
   * Returns a new tensor with a dimension of size one inserted at the specified position.
   * 
   * @param dim The index at which to insert the singleton dimension
   * @returns A new tensor with an additional dimension
   */
  unsqueeze(dim) {
    const newShape = [...this.shape];
    const actualDim = dim < 0 ? this.shape.length + dim + 1 : dim;
    if (actualDim < 0 || actualDim > this.shape.length) {
      throw new Error(`Dimension out of range (expected to be in range of [${-this.shape.length - 1}, ${this.shape.length}], but got ${dim})`);
    }
    newShape.splice(actualDim, 0, 1);
    const newTensor = new Tensor([...this._data], this.requires_grad, this.device);
    newTensor.shape = newShape;
    return newTensor;
  }
  slice(indices) {
    function extractSlice(data, currentDim) {
      if (!Array.isArray(data) || currentDim >= indices.length) {
        return data;
      }
      const index = indices[currentDim];
      if (index === null) {
        return data.map((item) => extractSlice(item, currentDim + 1));
      } else {
        return extractSlice(data[index], currentDim + 1);
      }
    }
    const slicedData = extractSlice(this._data, 0);
    return new Tensor(slicedData, this.requires_grad, this.device);
  }
  setSlice(indices, value) {
    function setSliceValues(data, valueData2, currentDim) {
      if (!Array.isArray(data) || currentDim >= indices.length) {
        return;
      }
      const index = indices[currentDim];
      if (index === null) {
        for (let i = 0; i < data.length; i++) {
          const nextValueData = Array.isArray(valueData2) && valueData2.length > i ? valueData2[i] : valueData2;
          if (currentDim === indices.length - 1) {
            data[i] = Array.isArray(nextValueData) ? [...nextValueData] : nextValueData;
          } else {
            setSliceValues(data[i], nextValueData, currentDim + 1);
          }
        }
      } else {
        if (currentDim === indices.length - 1) {
          data[index] = Array.isArray(valueData2) ? [...valueData2] : valueData2;
        } else {
          setSliceValues(data[index], valueData2, currentDim + 1);
        }
      }
    }
    const valueData = value instanceof Tensor ? value._data : value;
    setSliceValues(this._data, valueData, 0);
    return this;
  }
  /**
   * Gets the sum of the Tensor over a specified dimension.
   * @param {number} dim - Dimension to sum over.
   * @param {boolean} keepdims - Whether to keep dimensions of original tensor.
   * @returns {Tensor} - Final tensor.
   */
  sum(dim = -1, keepdims = false) {
    const operation = new Sum();
    return operation.forward(this, dim, keepdims);
  }
  /**
   * Gets the mean of the Tensor over a specified dimension.
   * @param {number} dim - Dimension to get mean over.
   * @param {boolean} keepdims - Whether to keep dimensions of original tensor.
   * @returns {Tensor} - Final tensor.
   */
  mean(dim = -1, keepdims = false) {
    const operation = new Mean();
    return operation.forward(this, dim, keepdims);
  }
  /**
   * Gets the variance of the Tensor over a specified dimension.
   * @param {number} dim - Dimension to get variance over.
   * @param {boolean} keepdims - Whether to keep dimensions of original tensor.
   * @returns {Tensor} - Final tensor.
   */
  variance(dim = -1, keepdims = false) {
    const operation = new Variance();
    return operation.forward(this, dim, keepdims);
  }
  /**
   * Add integer or other Tensor to this Tensor.
   * @param {any} other - Tensor or integer to be added to this Tensor.
   * @returns {object} New tensor.
   */
  add(other) {
    const operation = new Add();
    return operation.forward(this, other);
  }
  /**
   * Subtract integer or other Tensor from this Tensor.
   * @param {any} other - Tensor or integer to be subtracted from this Tensor.
   * @returns {object} New tensor.
   */
  sub(other) {
    if (typeof other === "number") {
      return this.add(-other);
    } else if (other instanceof Tensor) {
      return this.add(other.neg());
    } else {
      throw Error('Argument "other" is not a Tensor or a number.');
    }
  }
  /**
   * Get element-wise opposite of given tensor ( every element * (-1) )
   * @returns {object} New tensor.
   */
  neg() {
    const operation = new Neg();
    return operation.forward(this);
  }
  /**
   * Multiply this Tensor by integer or other Tensor.
   * @param {any} other - Tensor or integer to multiply this Tensor by.
   * @returns {object} New tensor.
   */
  mul(other) {
    const operation = new Mul();
    return operation.forward(this, other);
  }
  /**
   * Divide this Tensor by integer or other Tensor.
   * @param {Tensor | number} other - Tensor or integer to divide this Tensor by.
   * @returns {Tensor} New tensor.
   */
  div(other) {
    const operation = new Div();
    return operation.forward(this, other);
  }
  /**
   * Multiply this Tensor by integer or other Tensor.
   * @param {Tensor | number} other - Tensor or integer to multiply this Tensor by.
   * @returns {Tensor} New tensor.
   */
  matmul(other) {
    const operation = new MatMul();
    let device;
    if (this.device === "gpu" || other.device === "gpu") {
      device = "gpu";
    } else {
      device = "cpu";
    }
    if (other.forwardKernel === null || other.batch_size != this.shape.at(-2)) {
      if (device === "gpu") {
        const { GPU } = require$1("@eduardoleao052/gpu");
        if (other.batch_size != null) {
          other.batch_size = other.shape.at(-2);
          if (other.warned === false) {
            console.warn(
              "Testing batch size different from training batch size. JS-PyTorch recreating GPU Kernel (Less efficient)"
            );
            other.warned = true;
          }
        }
        other.gpu = new GPU();
        const kernelFunc = function(a, b, len) {
          let sum2 = 0;
          for (let i = 0; i < len; i++) {
            sum2 += a[this.thread.y][i] * b[i][this.thread.x];
          }
          return sum2;
        };
        other.forwardKernel = other.gpu.createKernel(kernelFunc, { loopMaxIterations: other.shape.at(-2) }).setOutput([other.shape.at(-1), this.shape.at(-2)]);
        other.backwardKernelA = other.gpu.createKernel(kernelFunc, { loopMaxIterations: other.shape.at(-1) }).setOutput([this.shape.at(-1), this.shape.at(-2)]);
        other.backwardKernelB = other.gpu.createKernel(kernelFunc, { loopMaxIterations: this.shape.at(-2) }).setOutput([other.shape.at(-1), other.shape.at(-2)]);
      } else {
        const kernelFunc = function(a, b, len) {
          const out = Array(a.length).fill(0).map(() => Array(b[0].length).fill(0));
          for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < b[0].length; j++) {
              let currentIndex = 0;
              for (let k = 0; k < len; k++) {
                currentIndex += a[i][k] * b[k][j];
              }
              out[i][j] = currentIndex;
            }
          }
          return out;
        };
        other.forwardKernel = kernelFunc;
        other.backwardKernelA = kernelFunc;
        other.backwardKernelB = kernelFunc;
      }
    }
    other.batch_size = this.shape.at(-2);
    return operation.forward(this, other);
  }
  /**
   * Get tensor to element-wise power of n.
   * @param {number} n - Exponent.
   * @returns {object} New tensor.
   */
  pow(n) {
    const operation = new Pow();
    return operation.forward(this, n);
  }
  /**
   * Get element-wise square root of given tensor.
   * @returns {object} New tensor.
   */
  sqrt() {
    const operation = new Sqrt();
    return operation.forward(this);
  }
  /**
   * Get element-wise exponentiation of given tensor ( e^(every element) )
   * @returns {object} New tensor.
   */
  exp() {
    const operation = new Exp();
    return operation.forward(this);
  }
  /**
   * Get element-wise natural log of given tensor ( ln(every element) )
   * @returns {object} New tensor.
   */
  log() {
    const operation = new Log();
    return operation.forward(this);
  }
  /**
   * Transpose the tensor along two consecutive dimensions:
   * @param {number} dim1 - First dimension.
   * @param {number} dim2 - Second dimension.
   * @returns {object} New tensor.
   */
  transpose(dim1, dim2) {
    const operation = new Transpose();
    return operation.forward(this, dim1, dim2);
  }
  /**
   * In a tensor, returns a list of elements in [index1], or [index1][index2];
   * @param {object} index1 - List containing indexes to extract data from in first dimension.
   * @param {object} index2 - List containing indexes to extract data from in second dimension [OPTIONAL].
   * @returns {object} New tensor.
   * @example
   * let a = tensor([[1,1,2,3],
   *                 [6,7,8,9]])
   *
   * // Returns tensor([2,6,9]):
   * a.at([0,1,1], [2,0,3])
   *
   * // Returns tensor([[1,1,2,3],
   *                    [6,7,8,9],
   *                    [1,1,2,3]])
   * a.at([0,1,0])
   */
  at(index1, index2) {
    const operation = new At();
    return operation.forward(this, index1, index2);
  }
  /**
   * Where the "condition" function returns True in "mask" Tensor, the "value" will fill the "this" Tensor.
   * @param {Tensor} mask - "condition" will be applied in this tensor element-wise.
   * @param {function} condition - Function that returns True or False element-wise.
   * @param {number} value - Value to fill Tensor when condition is met.
   * @returns {object} New tensor.
   * @example
   * let a = tensor([[1,5,2,3],
   *                 [6,7,2,9]])
   *
   * // Returns tensor([[1,0,2,3],
   * //                 [0,0,2,0]])
   * a.masked_fill(mask, (el) => {return el > 3}, 0)
   */
  masked_fill(mask, condition, value) {
    const operation = new MaskedFill();
    return operation.forward(this, mask, condition, value);
  }
  /**
   * Reshape the tensor into the new shape:
   * @param {object} shape - New tensor's shape.
   * @returns {object} New tensor.
   */
  reshape(shape) {
    const operation = new Reshape();
    return operation.forward(this, shape);
  }
  //TODO: Utility functions for Conv2D / MaxPool from the MR by TaylorHawkes bd9f840574ba1564919b27685f2427de4c688ab2
  img2col(kernel_height, kernel_width, stride, padding) {
    const operation = new Img2Col();
    return operation.forward(
      this,
      kernel_height,
      kernel_width,
      stride,
      padding
    );
  }
  maxpool(kernel_size, stride) {
    const operation = new MaxPool();
    return operation.forward(this, kernel_size, stride);
  }
}
class Parameter extends Tensor {
  /**
   * Creates new Parameter (an instance of the Tensor class that always tracks gradients).
   * @param {object} data - Iterable containing the data to be stored in the Tensor.
   */
  constructor(data) {
    super(data, true);
  }
}
class Add {
  cache;
  /**
   * Add tensors or tensor and integers.
   * @param {any} a - First tensor or integer.
   * @param {any} b - Second tensor or integer.
   * @returns {object} New tensor.
   */
  forward(a, b) {
    this.cache = [a, b];
    const aData = getData(a);
    const bData = getData(b);
    const z = new Tensor(
      _add(aData, bData),
      // data;
      requiresGrad(a) || requiresGrad(b)
      // requires_grad;
    );
    if (a instanceof Tensor && requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
    }
    if (b instanceof Tensor && requiresGrad(b)) {
      z.parents.push(b);
      b.children.push(z);
    }
    z.operation = this;
    return z;
  }
  backward(dz, z) {
    const [a, b] = this.cache;
    if (requiresGrad(a)) {
      let da = dz;
      da = broadcast(da, a);
      a.backward(da, z);
    }
    if (requiresGrad(b)) {
      let db = dz;
      db = broadcast(db, b);
      b.backward(db, z);
    }
  }
}
class Neg {
  cache;
  /**
   * Get element-wise opposite of given tensor ( every element * (-1) )
   * @param {object} a - Tensor to be multiplied by -1.
   * @returns {object} New tensor.
   */
  forward(a) {
    this.cache = a;
    const z = new Tensor(
      _neg(a._data),
      // data;
      requiresGrad(a)
      // requires_grad;
    );
    if (requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
      z.operation = this;
    }
    return z;
  }
  backward(dz, z) {
    const a = this.cache;
    if (requiresGrad(a)) {
      const da = neg(dz);
      a.backward(da, z);
    }
  }
}
class Mul {
  cache;
  /**
   * Perform element-wise multiplication between Tensors and integers or other Tensors.
   * @param {any} a - First tensor or integer.
   * @param {any} b - Second tensor or integer.
   * @returns {object} New tensor.
   */
  forward(a, b) {
    this.cache = [a, b];
    const aData = getData(a);
    const bData = getData(b);
    const z = new Tensor(
      _mul(aData, bData),
      // data;
      requiresGrad(a) || requiresGrad(b)
      // requires_grad;
    );
    if (a instanceof Tensor && requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
    }
    if (b instanceof Tensor && requiresGrad(b)) {
      z.parents.push(b);
      b.children.push(z);
    }
    z.operation = this;
    return z;
  }
  backward(dz, z) {
    const [a, b] = this.cache;
    if (requiresGrad(a)) {
      let da = new Tensor(_mul(dz.data, getData(b)));
      da = broadcast(da, a);
      a.backward(da, z);
    }
    if (requiresGrad(b)) {
      let db = new Tensor(_mul(dz.data, getData(a)));
      db = broadcast(db, b);
      b.backward(db, z);
    }
  }
}
class Div {
  cache;
  /**
   * Perform element-wise division between Tensors and integers or other Tensors.
   * @param {any} a - First tensor or integer.
   * @param {any} b - Second tensor or integer.
   * @returns {object} New tensor.
   */
  forward(a, b) {
    this.cache = [a, b];
    const aData = getData(a);
    const bData = getData(b);
    const z = new Tensor(
      _div(aData, bData),
      // data;
      requiresGrad(a) || requiresGrad(b)
      // requires_grad;
    );
    if (a instanceof Tensor && requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
    }
    if (b instanceof Tensor && requiresGrad(b)) {
      z.parents.push(b);
      b.children.push(z);
    }
    z.operation = this;
    return z;
  }
  backward(dz, z) {
    const [a, b] = this.cache;
    if (requiresGrad(a)) {
      let da = new Tensor(_mul(dz.data, _div(1, getData(b))));
      da = broadcast(da, a);
      a.backward(da, z);
    }
    if (requiresGrad(b)) {
      let db = new Tensor(
        _mul(dz.data, _neg(_div(getData(a), _pow(getData(b), 2))))
      );
      db = broadcast(db, b);
      b.backward(db, z);
    }
  }
}
class MatMul {
  cache;
  kernelFunc;
  thread;
  forward(a, b) {
    this.cache = [a, b];
    let aData = a.data;
    let bData = b.data;
    if (a.shape.length < b.shape.length) {
      aData = broadcastUp(aData, bData);
    } else {
      bData = broadcastUp(bData, aData);
    }
    const z = new Tensor(
      _matmul(aData, bData, b.forwardKernel),
      // data;
      requiresGrad(a) || requiresGrad(b)
      // requires_grad;
    );
    if (a instanceof Tensor && requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
    }
    if (b instanceof Tensor && requiresGrad(b)) {
      z.parents.push(b);
      b.children.push(z);
    }
    z.operation = this;
    return z;
  }
  backward(dz, z) {
    const [a, b] = this.cache;
    if (requiresGrad(a)) {
      const dzData = dz.data;
      let b_T = _transpose(b.data, b.ndims - 2);
      b_T = broadcastUp(b_T, dzData);
      let da = new Tensor(_matmul(dzData, b_T, b.backwardKernelA));
      da = broadcast(da, a);
      a.backward(da, z);
    }
    if (requiresGrad(b)) {
      const dzData = dz.data;
      let a_T = _transpose(a.data, a.ndims - 2);
      a_T = broadcastUp(a_T, dzData);
      let db = new Tensor(_matmul(a_T, dzData, b.backwardKernelB));
      db = broadcast(db, b);
      b.backward(db, z);
    }
  }
}
class Pow {
  cache;
  /**
   * Get tensor to element-wise power of n.
   * @param {object} a - Tensor to be elevated to the power of n.
   * @param {number} n - Exponent.
   * @returns {object} New tensor.
   */
  forward(a, n) {
    this.cache = a;
    const z = new Tensor(
      _pow(getData(a), n),
      // data;
      requiresGrad(a)
      // requires_grad;
    );
    if (requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
      z.operation = this;
    }
    return z;
  }
  backward(dz, z) {
    const a = this.cache;
    if (requiresGrad(a)) {
      const da = new Tensor(_mul(2, _mul(a.data, dz.data)));
      a.backward(da, z);
    }
  }
}
class Sqrt {
  cache;
  /**
   * Get element-wise square root of given tensor
   * @param {object} a - Tensor to be square rooted.
   * @returns {object} New tensor.
   */
  forward(a) {
    this.cache = a;
    const z = new Tensor(
      _sqrt(a._data),
      // data;
      requiresGrad(a)
      // requires_grad;
    );
    if (requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
      z.operation = this;
    }
    return z;
  }
  backward(dz, z) {
    const a = this.cache;
    if (requiresGrad(a)) {
      const da = new Tensor(
        _mul(_mul(_div(1, 2), _div(1, _sqrt(a.data))), dz.data)
      );
      a.backward(da, z);
    }
  }
}
class Exp {
  cache;
  /**
   * Get element-wise exponentiation of given tensor ( e^(every element) )
   * @param {object} a - Tensor to be exponentiated.
   * @returns {object} New tensor.
   */
  forward(a) {
    this.cache = a;
    const z = new Tensor(
      _exp(a._data),
      // data;
      requiresGrad(a)
      // requires_grad;
    );
    if (requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
      z.operation = this;
    }
    return z;
  }
  backward(dz, z) {
    const a = this.cache;
    if (requiresGrad(a)) {
      const da = new Tensor(_mul(_exp(a.data), dz.data));
      a.backward(da, z);
    }
  }
}
class Log {
  cache;
  /**
   * Get element-wise natural log of given tensor ( ln(every element) )
   * @param {object} a - Tensor we will take the log of.
   * @returns {object} New tensor.
   */
  forward(a) {
    this.cache = a;
    const z = new Tensor(
      _log(a._data),
      // data;
      requiresGrad(a)
      // requires_grad;
    );
    if (requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
      z.operation = this;
    }
    return z;
  }
  backward(dz, z) {
    const a = this.cache;
    if (requiresGrad(a)) {
      const da = new Tensor(_mul(_div(1, a.data), dz.data));
      a.backward(da, z);
    }
  }
}
class Sum {
  cache;
  /**
   * Gets the sum of a Tensor over a specified dimension.
   * @param {Tensor} a - Tensor to sum.
   * @param {dim} dim - Dimension to sum over.
   * @param {keepdims} keepdims - Whether to keep dimensions of original tensor.
   * @returns {Tensor} - Final tensor.
   */
  forward(a, dim, keepdims = false) {
    this.cache = [a, dim, keepdims];
    if (dim < 0) {
      dim = a.shape.length + dim;
    }
    if (dim >= a.shape.length) {
      throw Error("Dimension larger than array.");
    }
    const z = new Tensor(
      _sum(a._data, dim, keepdims),
      // New data.
      requiresGrad(a)
      // requires_grad.
    );
    if (requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
      z.operation = this;
    }
    return z;
  }
  backward(dz, z) {
    const [a, dim, keepdims] = this.cache;
    if (requiresGrad(a)) {
      if (keepdims) {
        dz = dz.sum(dim);
      }
      const da = broadcast(dz, a);
      a.backward(da, z);
    }
  }
}
class Mean {
  cache;
  /**
   * Gets the mean of a Tensor over a specified dimension.
   * @param {Tensor} a - Tensor to get mean from.
   * @param {dim} dim - Dimension to get mean over.
   * @param {keepdims} keepdims - Whether to keep dimensions of original tensor.
   * @returns {Tensor} - Final tensor.
   */
  forward(a, dim, keepdims = false) {
    if (dim < 0) {
      dim = a.shape.length + dim;
    }
    if (dim >= a.shape.length) {
      throw Error("Dimension larger than array.");
    }
    this.cache = [a, dim];
    const z = new Tensor(
      _mean(a._data, dim, keepdims),
      // New data.
      requiresGrad(a)
      // keep_dims.
    );
    if (requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
      z.operation = this;
    }
    return z;
  }
  backward(dz, z) {
    const [a, dim] = this.cache;
    if (requiresGrad(a)) {
      let da = new Tensor(_div(dz.data, a.shape[dim]));
      da = broadcast(da, a);
      a.backward(da, z);
    }
  }
}
class Variance {
  cache;
  /**
   * Gets the variance of a Tensor over a specified dimension.
   * @param {Tensor} a - Tensor to get variance of.
   * @param {dim} dim - Dimension to get variance over.
   * @param {keepdims} keepdims - Whether to keep dimensions of original tensor.
   * @returns {Tensor} - Final tensor.
   */
  forward(a, dim, keepdims = false) {
    if (dim < 0) {
      dim = a.shape.length + dim;
    }
    if (dim >= a.shape.length) {
      throw Error("Dimension larger than array.");
    }
    this.cache = [a, dim];
    const z = new Tensor(
      _variance(a._data, dim, keepdims),
      // New data.
      requiresGrad(a)
      // keep_dims.
    );
    if (requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
      z.operation = this;
    }
    return z;
  }
  backward(dz, z) {
    const [a, dim] = this.cache;
    if (requiresGrad(a)) {
      dz = broadcast(dz, a);
      const err = _add(a._data, _neg(_mean(a._data, dim, true)));
      const var_err = _mul(_mul(dz._data, 2), err);
      let da = _div(var_err, a.shape[dim]);
      da = new Tensor(da);
      a.backward(da, z);
    }
  }
}
class Transpose {
  cache;
  /**
   * Transpose the tensor along two consecutive dimensions:
   * @param {object} a - Tensor to transpose.
   * @param {number} dim1 - First dimension.
   * @param {number} dim2 - Second dimension.
   * @returns {object} New tensor.
   */
  forward(a, dimA, dimB) {
    this.cache = [a, dimA, dimB];
    if (dimA < 0) {
      dimA = a.shape.length + dimA;
    }
    if (dimB < 0) {
      dimB = a.shape.length + dimB;
    }
    let dim;
    if (dimB < dimA) {
      dim = dimB;
    } else if (dimB > dimA) {
      dim = dimA;
    } else {
      throw new Error("ValueError: dimensions are not consecutive.");
    }
    const z = new Tensor(
      _transpose(a._data, dim),
      // data;
      requiresGrad(a)
      // requires_grad;
    );
    if (requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
      z.operation = this;
    }
    return z;
  }
  backward(dz, z) {
    const [a, dimA, dimB] = this.cache;
    if (requiresGrad(a)) {
      const da = dz.transpose(dimA, dimB);
      a.backward(da, z);
    }
  }
}
class At {
  cache;
  forward(a, idx1, idx2 = null) {
    if (idx1) {
      idx1 = assureArray(idx1).flat(Infinity);
    }
    if (idx2) {
      idx2 = assureArray(idx2).flat(Infinity);
    }
    this.cache = [a, idx1, idx2];
    const z = new Tensor(
      _at(a._data, idx1, idx2),
      // data;
      requiresGrad(a)
      // requires_grad;
    );
    if (requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
      z.operation = this;
    }
    return z;
  }
  backward(dz, z) {
    const [a, idx1, idx2] = this.cache;
    if (requiresGrad(a)) {
      const da = zeros(a.shape);
      for (let i = 0; i < dz.length; i++) {
        if (idx2 != null) {
          da._data[idx1[i]][idx2[i]] = _add(
            da._data[idx1[i]][idx2[i]],
            dz._data[i]
          );
        } else {
          da._data[idx1[i]] = _add(da._data[idx1[i]], dz._data[i]);
        }
      }
      a.backward(da, z);
    }
  }
}
class MaskedFill {
  cache;
  forward(a, mask, condition, value) {
    this.cache = [a, mask, condition];
    const z = new Tensor(
      _masked_fill(a._data, mask._data, condition, value),
      // data;
      requiresGrad(a)
      // requires_grad;
    );
    if (requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
      z.operation = this;
    }
    return z;
  }
  backward(dz, z) {
    const [a, mask, condition] = this.cache;
    if (requiresGrad(a)) {
      const da = new Tensor(_masked_fill(dz._data, mask._data, condition, 0));
      a.backward(da, z);
    }
  }
}
class Reshape {
  cache;
  forward(a, shape) {
    this.cache = a;
    const z = new Tensor(
      _reshape(a._data, shape),
      // data;
      requiresGrad(a)
      // requires_grad;
    );
    if (requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
      z.operation = this;
    }
    return z;
  }
  backward(dz, z) {
    const a = this.cache;
    if (requiresGrad(a)) {
      const da = new Tensor(_reshape(dz.data, a.shape));
      a.backward(da, z);
    }
  }
}
class MaxPool {
  cache;
  forward(a, kernel_size, stride) {
    const [batch, channels, height, width] = a.shape;
    const [kh, kw] = kernel_size;
    const [sh, sw] = stride;
    const out_height = Math.floor((height - kh) / sh + 1);
    const out_width = Math.floor((width - kw) / sw + 1);
    const outputData = new Array(batch).fill(0).map(
      () => new Array(channels).fill(0).map(
        () => new Array(out_height).fill(0).map(() => new Array(out_width).fill(0))
      )
    );
    const maxIndices = new Array(batch).fill(0).map(
      () => new Array(channels).fill(0).map(
        () => new Array(out_height).fill(0).map(() => new Array(out_width).fill([0, 0]))
      )
    );
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let i = 0; i < out_height; i++) {
          for (let j = 0; j < out_width; j++) {
            const h_start = i * sh;
            const w_start = j * sw;
            const h_end = h_start + kh;
            const w_end = w_start + kw;
            let max_val = -Infinity;
            let max_idx = [0, 0];
            for (let ki = h_start; ki < h_end; ki++) {
              for (let kj = w_start; kj < w_end; kj++) {
                if (ki >= 0 && ki < height && kj >= 0 && kj < width) {
                  const val = a.data[b][c][ki][kj];
                  if (val > max_val) {
                    max_val = val;
                    max_idx = [ki - h_start, kj - w_start];
                  }
                }
              }
            }
            outputData[b][c][i][j] = max_val;
            maxIndices[b][c][i][j] = max_idx;
          }
        }
      }
    }
    this.cache = { x: a, maxIndices, kernel_size, stride };
    const z = new Tensor(outputData, requiresGrad(a));
    if (a instanceof Tensor && requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
    }
    z.operation = this;
    return z;
  }
  backward(dz, z) {
    const { x, maxIndices, kernel_size, stride } = this.cache;
    const [sh, sw] = stride;
    const [batch, channels, out_height, out_width] = dz.shape;
    const dx = new Array(batch).fill(0).map(
      () => new Array(channels).fill(0).map(
        () => new Array(x.shape[2]).fill(0).map(() => new Array(x.shape[3]).fill(0))
      )
    );
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let i = 0; i < out_height; i++) {
          for (let j = 0; j < out_width; j++) {
            const [h_idx, w_idx] = maxIndices[b][c][i][j];
            const h_start = i * sh;
            const w_start = j * sw;
            dx[b][c][h_start + h_idx][w_start + w_idx] += dz.data[b][c][i][j];
          }
        }
      }
    }
    if (x.requires_grad) {
      const dxTensor = new Tensor(dx);
      x.backward(dxTensor, z);
    }
  }
}
class Img2Col {
  cache;
  forward(a, kernel_height, kernel_width, stride, padding) {
    this.cache = [a, kernel_height, kernel_width, stride, padding];
    const [batch, channels, height, width] = a.shape;
    const out_height = Math.floor((height + 2 * padding[0] - kernel_height) / stride[0]) + 1;
    const out_width = Math.floor((width + 2 * padding[1] - kernel_width) / stride[1]) + 1;
    const col_data = [];
    for (let b = 0; b < batch; b++) {
      for (let i = 0; i < out_height; i++) {
        for (let j = 0; j < out_width; j++) {
          const patch = [];
          for (let c = 0; c < channels; c++) {
            for (let kh = 0; kh < kernel_height; kh++) {
              for (let kw = 0; kw < kernel_width; kw++) {
                const h_idx = i * stride[0] - padding[0] + kh;
                const w_idx = j * stride[1] - padding[1] + kw;
                if (h_idx >= 0 && h_idx < height && w_idx >= 0 && w_idx < width) {
                  patch.push(a.data[b][c][h_idx][w_idx]);
                } else {
                  patch.push(0);
                }
              }
            }
          }
          col_data.push(patch);
        }
      }
    }
    const z = new Tensor(col_data, requiresGrad(a));
    if (a instanceof Tensor && requiresGrad(a)) {
      z.parents.push(a);
      a.children.push(z);
    }
    z.operation = this;
    return z;
  }
  backward(dz, z) {
    const [a, kernel_height, kernel_width, stride, padding] = this.cache;
    const [batch, channels, height, width] = a.shape;
    const out_height = Math.floor((height + 2 * padding[0] - kernel_height) / stride[0]) + 1;
    const out_width = Math.floor((width + 2 * padding[1] - kernel_width) / stride[1]) + 1;
    const dx = new Tensor(
      new Array(batch).fill(0).map(
        () => new Array(channels).fill(0).map(
          () => new Array(height).fill(0).map(() => new Array(width).fill(0))
        )
      )
    );
    let col_index = 0;
    for (let b = 0; b < batch; b++) {
      for (let i = 0; i < out_height; i++) {
        for (let j = 0; j < out_width; j++) {
          const gradient_patch = dz.data[col_index];
          let patch_index = 0;
          for (let c = 0; c < channels; c++) {
            for (let kh = 0; kh < kernel_height; kh++) {
              for (let kw = 0; kw < kernel_width; kw++) {
                const h_idx = i * stride[0] - padding[0] + kh;
                const w_idx = j * stride[1] - padding[1] + kw;
                if (h_idx >= 0 && h_idx < height && w_idx >= 0 && w_idx < width) {
                  dx.data[b][c][h_idx][w_idx] += gradient_patch[patch_index];
                }
                patch_index++;
              }
            }
          }
          col_index++;
        }
      }
    }
    if (a.requires_grad) {
      a.backward(dx, z);
    }
  }
}
function mean(a, dim = -1, keepdims = false) {
  return a.mean(dim, keepdims);
}
function variance(a, dim = -1, keepdims = false) {
  return a.variance(dim, keepdims);
}
function add(a, b) {
  return a.add(b);
}
function neg(a) {
  return a.neg();
}
function mul(a, b) {
  return a.mul(b);
}
function div(a, b) {
  const operation = new Div();
  return operation.forward(a, b);
}
function pow(a, n) {
  const operation = new Pow();
  return operation.forward(a, n);
}
function sqrt(a) {
  return a.sqrt();
}
function exp(a) {
  return a.exp();
}
function log(a) {
  return a.log();
}
function matmul(a, b) {
  return a.matmul(b);
}
function transpose(a, dim1, dim2) {
  return a.transpose(dim1, dim2);
}
function at(a, idx1, idx2) {
  return a.at(idx1, idx2);
}
function masked_fill(a, mask, condition, value) {
  return a.masked_fill(mask, condition, value);
}
function reshape(a, shape) {
  return a.reshape(shape);
}
function _sum(a, dim, keepdims) {
  if (dim == 0) {
    const sum2 = a.reduce((a2, b) => _add(a2, b), 0);
    if (keepdims) {
      return Array(a.length).fill(sum2);
    } else {
      return sum2;
    }
  } else if (typeof a === "object") {
    return a.map((element) => _sum(element, dim - 1, keepdims));
  } else {
    throw Error("Dimension invalid.");
  }
}
function _mean(a, dim, keepdims) {
  if (dim == 0) {
    const reduced = _div(
      a.reduce((a2, b) => _add(a2, b), 0),
      a.length
    );
    if (keepdims) {
      return Array(a.length).fill(reduced);
    } else {
      return reduced;
    }
  } else if (typeof a === "object") {
    return a.map((element) => _mean(
      element,
      dim - 1
      /*, keepdims*/
    ));
  } else {
    throw Error("Dimension invalid.");
  }
}
function _variance(a, dim, keepdims) {
  if (dim == 0) {
    const mean2 = _div(
      a.reduce((a2, b) => _add(a2, b), 0),
      a.length
    );
    const squares = a.map((el) => (el - mean2) ** 2);
    const variance2 = _div(
      squares.reduce((a2, b) => _add(a2, b), 0),
      a.length
    );
    if (keepdims) {
      return Array(a.length).fill(variance2);
    } else {
      return variance2;
    }
  } else if (typeof a === "object") {
    return a.map((element) => _variance(
      element,
      dim - 1
      /*keepdims*/
    ));
  } else {
    throw Error("Dimension invalid.");
  }
}
function _add(a, b) {
  if (typeof a === "number" && typeof b === "number") {
    return a + b;
  } else if (typeof a === "number" && b instanceof Array) {
    return b.map((element) => _add(element, a));
  } else if (a instanceof Array && typeof b === "number") {
    return a.map((element) => _add(element, b));
  } else if (a instanceof Array && b instanceof Array) {
    const aShape = getShape(a);
    const bShape = getShape(b);
    if (JSON.stringify(aShape) === JSON.stringify(bShape)) {
      return a.map((element, idx) => _add(element, b[idx]));
    } else if (aShape.length > bShape.length) {
      let idx;
      for (let i = 0; i < aShape.length; i++) {
        if (JSON.stringify(aShape.slice(i, i + bShape.length)) === JSON.stringify(bShape)) {
          idx = i;
        }
      }
      if (idx === 0) {
        return a.map((element, idx2) => _add(element, b[idx2]));
      } else {
        return a.map((element) => _add(element, b));
      }
    } else if (aShape.length < bShape.length) {
      let idx;
      for (let i = 0; i < bShape.length; i++) {
        if (JSON.stringify(bShape.slice(i, i + aShape.length)) === JSON.stringify(aShape)) {
          idx = i;
        }
      }
      if (idx === 0) {
        return b.map((element, idx2) => _add(a[idx2], element));
      } else {
        return b.map((element) => _add(a, element));
      }
    } else {
      throw Error("Given arguments cannot be added.");
    }
  } else {
    throw Error("Given arguments cannot be added.");
  }
}
function _neg(a) {
  if (typeof a === "number") {
    return -a;
  } else if (typeof a === "object") {
    return a.map((element) => _neg(element));
  } else {
    throw new TypeError("the input data is not a number.");
  }
}
function _mul(a, b) {
  if (typeof a === "number" && typeof b === "number") {
    return a * b;
  } else if (typeof a === "number" && b instanceof Array) {
    return b.map((element) => _mul(element, a));
  } else if (a instanceof Array && typeof b === "number") {
    return a.map((element) => _mul(element, b));
  } else if (a instanceof Array && b instanceof Array) {
    const aShape = getShape(a);
    const bShape = getShape(b);
    if (JSON.stringify(aShape) === JSON.stringify(bShape)) {
      return a.map((element, idx) => _mul(element, b[idx]));
    } else if (aShape.length > bShape.length) {
      let idx;
      for (let i = 0; i < aShape.length; i++) {
        if (JSON.stringify(aShape.slice(i, i + bShape.length)) === JSON.stringify(bShape)) {
          idx = i;
        }
      }
      if (idx === 0) {
        return a.map((element, idx2) => _mul(element, b[idx2]));
      } else {
        return a.map((element) => _mul(element, b));
      }
    } else if (aShape.length < bShape.length) {
      let idx;
      for (let i = 0; i < bShape.length; i++) {
        if (JSON.stringify(bShape.slice(i, i + aShape.length)) === JSON.stringify(aShape)) {
          idx = i;
        }
      }
      if (idx === 0) {
        return b.map((element, idx2) => _mul(a[idx2], element));
      } else {
        return b.map((element) => _mul(a, element));
      }
    }
  }
}
function _div(a, b) {
  if (typeof a === "number" && typeof b === "number") {
    return a / b;
  } else if (typeof a === "number" && b instanceof Array) {
    return b.map((element) => _div(a, element));
  } else if (a instanceof Array && typeof b === "number") {
    return a.map((element) => _div(element, b));
  } else if (a instanceof Array && b instanceof Array) {
    const aShape = getShape(a);
    const bShape = getShape(b);
    if (JSON.stringify(aShape) === JSON.stringify(bShape)) {
      return a.map((element, idx) => _div(element, b[idx]));
    } else if (aShape.length > bShape.length) {
      let idx;
      for (let i = 0; i < aShape.length; i++) {
        if (JSON.stringify(aShape.slice(i, i + bShape.length)) === JSON.stringify(bShape)) {
          idx = i;
        }
      }
      if (idx === 0) {
        return a.map((element, idx2) => _div(element, b[idx2]));
      } else {
        return a.map((element) => _div(element, b));
      }
    } else if (aShape.length < bShape.length) {
      let idx;
      for (let i = 0; i < bShape.length; i++) {
        if (JSON.stringify(bShape.slice(i, i + aShape.length)) === JSON.stringify(aShape)) {
          idx = i;
        }
      }
      if (idx === 0) {
        return b.map((element, idx2) => _div(a[idx2], element));
      } else {
        return b.map((element) => _div(a, element));
      }
    }
  }
}
function _matmul(a, b, kernel) {
  if (typeof a === "number") {
    throw new Error("Cannot perform MatMul with given shapes.");
  }
  if (typeof a[0][0] === "object") {
    return a.map(
      (element, idx) => _matmul(element, b[idx], kernel)
    );
  } else {
    if (a[0].length === b.length && typeof a[0][0] === "number") {
      let out = kernel(a, b, b.length);
      out = out.map((el) => Array.from(el));
      return out;
    } else {
      throw Error(
        `Cannot perform Matrix Multiplication: cannot broadcast ${[
          a.length,
          a[0].length
        ]} and ${[b.length, b[0].length]}`
      );
    }
  }
}
function _pow(a, n) {
  let z = a;
  for (let i = 0; i < n - 1; i++) {
    z = _mul(z, a);
  }
  return z;
}
function _sqrt(a) {
  if (typeof a === "number") {
    return Math.sqrt(a);
  } else if (a instanceof Array) {
    return a.map((element) => _sqrt(element));
  } else {
    throw new TypeError("the input data is not a number.");
  }
}
function _exp(a) {
  if (typeof a === "number") {
    return 2.718281828459045 ** a;
  } else if (a instanceof Array) {
    return a.map((element) => _exp(element));
  } else {
    throw new TypeError("the input data is not a number.");
  }
}
function _log(a) {
  if (typeof a === "number") {
    return Math.log(a);
  } else if (a instanceof Array) {
    return a.map((element) => _log(element));
  } else {
    throw new TypeError("the input data is not a number.");
  }
}
function _transpose(a, dim) {
  if (dim == 0) {
    const newArray = Array(a[0].length).fill(0).map(() => Array(a.length).fill(0));
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < a[i].length; j++) {
        newArray[j][i] = a[i][j];
      }
    }
    return newArray;
  } else if (a instanceof Array) {
    return a.map((element) => _transpose(element, dim - 1));
  } else {
    throw Error("ValueError: dimensions are invalid.");
  }
}
function _at(a, idx1, idx2) {
  if (idx2) {
    return Array(idx1.length).fill(0).map((_, i) => a[idx1[i]][idx2[i]]);
  } else {
    return Array(idx1.length).fill(0).map((_, i) => a[idx1[i]]);
  }
}
function _masked_fill(a, mask, condition, value) {
  if (typeof mask === "number") {
    if (typeof a != "number") {
      throw new Error("Tensor and Mask not broadcastable");
    }
    if (condition(mask)) {
      return value;
    } else {
      return a;
    }
  } else if (typeof a === "object") {
    return a.map(
      (element, idx) => _masked_fill(element, mask[idx], condition, value)
    );
  } else {
    throw new Error("The input data is not a number.");
  }
}
function _reshape(a, shape) {
  if (getShape(a).reduce((a2, b) => a2 * b, 1) != shape.reduce((a2, b) => a2 * b, 1)) {
    throw new Error("Attempting to reshape into invalid shape.");
  }
  function _build(a2, shape2, idx, numberOfEls) {
    if (shape2.length > 1) {
      const emptyArray = Array(shape2[0]).fill(0);
      let offSet = idx;
      numberOfEls = numberOfEls / shape2[0];
      const myArray = emptyArray.map(
        (_, idx2) => _build(a2, shape2.slice(1), offSet + idx2 * numberOfEls, numberOfEls)
      );
      return myArray;
    } else {
      const myArray = a2.slice(idx, idx + numberOfEls);
      return myArray;
    }
  }
  const flat = a.flat(Infinity);
  const built = _build(flat, shape, 0, flat.length);
  return built;
}
function _tensorInitializer(shape, valueFunc) {
  if (shape.length === 1) {
    const emptyArray = Array(shape[0]).fill(0);
    return emptyArray.map(() => valueFunc());
  } else {
    const currentSize = shape[0];
    const emptyArray = Array(currentSize).fill(0);
    return emptyArray.map(() => _tensorInitializer(shape.slice(1), valueFunc));
  }
}
function tensor(data, requires_grad = false, device = "cpu") {
  return new Tensor(data, requires_grad, device);
}
function zeros(shape, requires_grad = false, device = "cpu") {
  return new Tensor(
    _tensorInitializer(shape, () => 0),
    requires_grad,
    device
  );
}
function ones(shape, requires_grad = false, device = "cpu") {
  return new Tensor(
    _tensorInitializer(shape, () => 1),
    requires_grad,
    device
  );
}
function tril(shape, requires_grad = false, device = "cpu") {
  const z = ones(shape, requires_grad);
  for (let i = 0; i < shape[0]; i++) {
    for (let j = 0; j < shape[0]; j++) {
      if (j > i) {
        z._data[i][j] = 0;
      }
    }
  }
  return new Tensor(z._data, requires_grad, device);
}
function rand(shape, requires_grad = false, device = "cpu") {
  return new Tensor(
    _tensorInitializer(shape, () => Math.random()),
    requires_grad,
    device
  );
}
function randn(shape, requires_grad = false, device = "cpu", xavier = false) {
  return new Tensor(
    _tensorInitializer(shape, () => {
      const mean2 = Math.random() * 0.98 + 1e-3;
      const variance2 = Math.random() * 0.98 + 1e-3;
      const num = Math.sqrt(-2 * Math.log(mean2)) * Math.cos(2 * Math.PI * variance2);
      if (xavier) {
        return num / Math.sqrt(shape[0]);
      } else {
        return num;
      }
    }),
    requires_grad,
    device
  );
}
function randint(low = 0, high = 1, shape = [1], requires_grad = false) {
  return new Tensor(
    _tensorInitializer(shape, () => {
      return Math.floor(Math.random() * (high - low)) + low;
    }),
    requires_grad
  );
}
function requiresGrad(a) {
  if (a instanceof Tensor) {
    return a.requires_grad;
  } else {
    return false;
  }
}
function broadcast(a, b) {
  function _broadcast(out2, b2) {
    if (typeof out2 === "number" && typeof b2 === "number") {
      return out2;
    } else if (typeof out2 === "number" && b2 instanceof Array) {
      const newArray = Array(b2.length).fill(out2);
      return _broadcast(newArray, b2);
    } else if (out2 instanceof Array && typeof b2 === "number") {
      return _broadcast(_sum(out2, 0), b2);
    } else if (JSON.stringify(getShape(out2)) === JSON.stringify(getShape(b2))) {
      return out2;
    } else if (out2 instanceof Array && b2 instanceof Array) {
      const outShape = getShape(out2);
      const bShape = getShape(b2);
      if (outShape.length > bShape.length) {
        let idx;
        for (let i = 0; i < outShape.length; i++) {
          if (JSON.stringify(outShape.slice(i, i + bShape.length)) === JSON.stringify(bShape)) {
            idx = i;
          }
        }
        if (idx === 0) {
          return out2.map((element, idx2) => _broadcast(element, b2[idx2]));
        } else {
          return _sum(out2, 0);
        }
      } else if (outShape.length < bShape.length) {
        let idx;
        for (let i = 0; i < bShape.length; i++) {
          if (JSON.stringify(bShape.slice(i, i + outShape.length)) === JSON.stringify(outShape)) {
            idx = i;
          }
        }
        if (idx === 0) {
          return out2.map((element) => _broadcast(element, b2[0]));
        } else {
          return Array(b2.length).fill(0).map(() => _broadcast(out2, b2[0]));
        }
      } else {
        const _broadcastSideways = (out3, b3) => {
          if (out3 instanceof Array && b3.length != out3.length) {
            if (b3.length === 1) {
              return [_sum(out3, 0)];
            } else if (out3.length === 1) {
              const emptyArray = Array(b3.length).fill(zeros);
              return emptyArray.map(() => out3[0]);
            } else {
              throw Error(
                `Shapes ${getShape(out3)} and ${getShape(b3)} not broadcastable.`
              );
            }
          } else {
            if (out3 instanceof Array) {
              return out3.map(
                (element, idx) => _broadcastSideways(element, b3[idx])
              );
            } else if (typeof out3 === "number") {
              return [null].map(
                (element, idx) => _broadcastSideways(element, b3[idx])
              );
            } else {
              throw Error("Shapes not broadcastable.");
            }
          }
        };
        return _broadcastSideways(out2, b2);
      }
    } else {
      throw Error("Shapes not broadcastable.");
    }
  }
  let out = a.data;
  while (JSON.stringify(getShape(out)) != JSON.stringify(b.shape)) {
    out = assureArray(_broadcast(out, b.data));
  }
  return new Tensor(out);
}
function broadcastUp(inElement, outElement) {
  function _broadcastUp(inElement2, outElement2) {
    if (getShape(inElement2).length + 1 === getShape(outElement2).length) {
      const emptyArray = Array(outElement2.length).fill(zeros);
      return emptyArray.map(() => inElement2);
    } else {
      const emptyArray = Array(outElement2.length).fill(zeros);
      return emptyArray.map(
        (_, idx) => _broadcastUp(inElement2, outElement2[idx])
      );
    }
  }
  while (getShape(inElement).length < getShape(outElement).length) {
    inElement = _broadcastUp(inElement, outElement);
  }
  return inElement;
}
function argmax(input, dim = -1, keepdim = false) {
  const actualDim = dim < 0 ? input.shape.length + dim : dim;
  if (actualDim < 0 || actualDim >= input.shape.length) {
    throw new Error(`Dimension out of range (expected to be in range of [${-input.shape.length}, ${input.shape.length - 1}], but got ${dim})`);
  }
  const outputShape = [...input.shape];
  if (!keepdim) {
    outputShape.splice(actualDim, 1);
  } else {
    outputShape[actualDim] = 1;
  }
  const result = new Tensor(outputShape, false, input.device);
  function processSlice(data, indices = [], depth = 0) {
    if (depth === actualDim) {
      let maxIndex = 0;
      let maxValue = data[0];
      for (let i = 1; i < data.length; i++) {
        if (data[i] > maxValue) {
          maxValue = data[i];
          maxIndex = i;
        }
      }
      return maxIndex;
    }
    if (!Array.isArray(data)) {
      return data;
    }
    const results = [];
    for (let i = 0; i < data.length; i++) {
      const newIndices = [...indices, i];
      const sliceResult = processSlice(data[i], newIndices, depth + 1);
      results.push(sliceResult);
    }
    return results;
  }
  let resultData = processSlice(input.data);
  if (keepdim) {
    let addDimension2 = function(data, dim2, currentDim = 0) {
      if (!Array.isArray(data)) {
        return dim2 === currentDim ? [data] : data;
      }
      if (dim2 === currentDim) {
        return [data];
      }
      return data.map((item) => addDimension2(item, dim2, currentDim + 1));
    };
    resultData = addDimension2(resultData, actualDim);
  }
  result._data = resultData;
  return result;
}

class Module {
  // Instantiate Module's mode initially as "train":
  mode = "train";
  /**
   * Returns all model parameters in a list.
   * @returns {object} List with parameters in the model.
   */
  parameters() {
    let params = [];
    for (const [_, value] of this.entries()) {
      if (value instanceof Module) {
        params = params.concat(value.parameters());
      } else if (value instanceof Parameter) {
        params.push(value);
      } else if (value instanceof Tensor) {
        if (value.requires_grad) {
          params.push(value);
        }
      }
    }
    return params;
  }
  /**
   * Sets module's mode to train, which influences layers like Dropout
   */
  train() {
    this.mode = "train";
    for (const [_, param] of this.entries()) {
      if (param instanceof Module) {
        param.train();
      }
    }
  }
  /**
   * Sets module's mode to eval, which influences layers like Dropout
   */
  eval() {
    this.mode = "eval";
    for (const [_, param] of this.entries()) {
      if (param instanceof Module) {
        param.eval();
      }
    }
  }
  /**
   * Returns an array of key/values of the enumerable properties of the Module
   * @returns {object} List with parameters in the model.
   */
  entries() {
    return Object.entries(this);
  }
}
class Linear extends Module {
  W;
  b;
  has_bias;
  /**
   * Simple linear layer, with weight matrix and optional bias. Does not contain nonlinearity.
   *
   * @param {number} in_size - size of the last dimention of the input array.
   * @param {number} out_size - size of the last dimention of the output array.
   * @param {string} device - Device to perform Tensor operations. Either "gpu" or "cpu".
   * @param {boolean} bias - wether to include a bias term.
   * @param {boolean} xavier - Wether to use xavier initialization (divide by square root of first input dimension).
   */
  constructor(in_size, out_size, device = "cpu", bias = true, xavier = true) {
    super();
    this.W = randn([in_size, out_size], true, device, xavier);
    this.b = zeros([out_size], true);
    this.has_bias = bias;
  }
  /**
   * Performs forward pass through the Linear layer.
   * @param {Tensor} x - input Tensor.
   * @returns {Tensor} new Tensor. Out = (In @ W) + b.
   */
  forward(x) {
    let z = x.matmul(this.W);
    if (this.has_bias) {
      z = z.add(this.b);
    }
    return z;
  }
}
class MultiHeadSelfAttention extends Module {
  Wk;
  Wq;
  Wv;
  residual_proj;
  mask;
  att_dropout;
  residual_dropout;
  softmax;
  H;
  /**
   * Full transformer Layer implementation.
   *
   * @param {number} in_size - size of the last dimention of the input array.
   * @param {number} out_size - size of the last dimention of the output array.
   * @param {number} n_heads - number of parallel heads to be computed (must equally divide in_size).
   * @param {number} n_timesteps - length of text sequence to be processed bt Transformer.
   * @param {number} dropout_prob - probability of zeroing each activation in dropout Layer.
   * @param {string} device - Device to perform Tensor operations. Either "gpu" or "cpu".
   */
  constructor(in_size, out_size, n_heads, n_timesteps, dropout_prob = 0, device = "cpu") {
    super();
    this.Wk = new Linear(in_size, in_size, device, true, false);
    this.Wq = new Linear(in_size, in_size, device, true, false);
    this.Wv = new Linear(in_size, in_size, device, true, false);
    this.residual_proj = new Linear(in_size, out_size, device, true, false);
    this.mask = tril([n_timesteps, n_timesteps], false);
    this.att_dropout = new Dropout(dropout_prob);
    this.residual_dropout = new Dropout(dropout_prob);
    this.softmax = new Softmax();
    this.H = in_size / n_heads;
    if (in_size % n_heads != 0) {
      throw new Error("Embedding dimension not divisible in equal heads.");
    }
  }
  /**
   * Performs Multi Head Self-Attention on "x" tensor.
   * @param {Tensor} x - input Tensor.
   * @returns {Tensor} new Tensor.
   */
  forward(x) {
    const [B, T, D] = x.shape;
    const H = this.H;
    const nh = D / H;
    let k = this.Wk.forward(x);
    let q = this.Wq.forward(x);
    let v = this.Wv.forward(x);
    k = k.reshape([B, T, nh, H]).transpose(1, 2);
    q = q.reshape([B, T, nh, H]).transpose(1, 2);
    v = v.reshape([B, T, nh, H]).transpose(1, 2);
    const kT = k.transpose(-2, -1);
    let att = q.matmul(kT);
    att = att.div(H ** 2);
    const mask = broadcast(this.mask, att);
    att = att.masked_fill(mask, (el) => el === 0, -Infinity);
    att = this.softmax.forward(att, -1);
    att = this.att_dropout.forward(att);
    let out = att.matmul(v);
    out = out.transpose(1, 2).reshape([B, T, D]);
    out = this.residual_proj.forward(out);
    out = this.residual_dropout.forward(out);
    return out;
  }
}
class FullyConnected extends Module {
  l1;
  relu;
  l2;
  dropout;
  /**
   * Small block composed of two Linear layers, a ReLU non-linearity and a Dropout layer.
   *
   * @param {number} in_size - size of the last dimention of the input array.
   * @param {number} out_size - size of the last dimention of the output array.
   * @param {number} dropout_prob - probability of zeroing each activation in dropout Layer.
   * @param {string} device - Device to perform Tensor operations. Either "gpu" or "cpu".
   * @param {boolean} bias - wether to include a bias term.
   */
  constructor(in_size, out_size, dropout_prob = 0, device = "cpu", bias = true) {
    super();
    this.l1 = new Linear(in_size, in_size * 2, device, true, bias);
    this.relu = new ReLU();
    this.l2 = new Linear(in_size * 2, out_size);
    this.dropout = new Dropout(dropout_prob);
  }
  /**
   *  Passes "x" tensor through the Fully Connected layers.
   * @param {Tensor} x - input Tensor.
   * @returns {Tensor} new Tensor.
   */
  forward(x) {
    let z = this.l1.forward(x);
    z = this.relu.forward(z);
    z = this.l2.forward(z);
    z = this.dropout.forward(z);
    return z;
  }
}
class Block extends Module {
  att;
  ln1;
  fcc;
  ln2;
  /**
   * Full transformer decoder block. Composed of Multi Head Self Attention, Fully connected layers and Layer Norms.
   *
   * @param {number} in_size - size of the last dimention of the input array.
   * @param {number} out_size - size of the last dimention of the output array.
   * @param {number} n_heads - number of parallel heads to be computed (must equally divide in_size).
   * @param {number} n_timesteps - length of text sequence to be processed bt Transformer.
   * @param {number} dropout_prob - probability of zeroing each activation in dropout Layer.
   * @param {string} device - Device to perform Tensor operations. Either "gpu" or "cpu".
   */
  constructor(in_size, out_size, n_heads, n_timesteps, dropout_prob = 0, device = "cpu") {
    super();
    this.att = new MultiHeadSelfAttention(
      in_size,
      in_size,
      n_heads,
      n_timesteps,
      dropout_prob,
      device
    );
    this.ln1 = new LayerNorm(in_size);
    this.fcc = new FullyConnected(in_size, out_size, dropout_prob, device, true);
    this.ln2 = new LayerNorm(out_size);
  }
  /**
   * Passes "x" tensor through a full transformer Block.
   * @param {Tensor} x - input Tensor.
   * @returns {Tensor} new Tensor.
   */
  forward(x) {
    let z = x.add(this.att.forward(this.ln1.forward(x)));
    z = z.add(this.fcc.forward(this.ln2.forward(z)));
    return z;
  }
}
class Embedding extends Module {
  E;
  /**
   * Embedding class, turns indexes into vectors.
   *
   * @param {number} vocab_size - number of different indexes (vocabulary size).
   * @param {number} embed_size - size of the embedding vector generated.
   */
  constructor(vocab_size, embed_size) {
    super();
    this.E = randn([vocab_size, embed_size], true, "cpu", false);
  }
  /**
   * Extracts embedding from rows in "idx":
   * @param {Tensor} idx - rows to get embedding from.
   * @returns {Tensor} new Tensor. Out = (In @ W) + b.
   */
  forward(idx) {
    const [B, T] = idx.shape;
    let x = this.E.at(idx);
    x = x.reshape([B, T, this.E.shape[1]]);
    return x;
  }
}
class PositionalEmbedding extends Module {
  E;
  /**
   * Embedding class, turns indexes into vectors based on it's position through an optimized lookup table.
   *
   * @param {number} input_size - number of different embeddings (size of the input).
   * @param {number} embed_size - size of the embedding vector generated.
   */
  constructor(input_size, embed_size) {
    super();
    this.E = randn([input_size, embed_size], true, "cpu", false);
  }
  /**
   * Gets embedding for timesteps in "idx" array.
   * @param {object} idx - Array [Batch x Timesteps]. Timesteps will be filled with positional embeddings.
   * @returns {Tensor} new Tensor.
   */
  forward(idx) {
    const [_, T] = idx.shape;
    const x = this.E.at([...Array(T).keys()]);
    return x;
  }
}
class ReLU extends Module {
  /**
   * Rectified Linear Unit nonlinearity. Returns z if z>0 else 0.
   */
  constructor() {
    super();
  }
  /**
   * Performs forward pass through Rectified Linear Unit nonlinearity. Returns z if z>0 else 0.
   * @param {Tensor} z - input Tensor.
   * @returns {Tensor} new Tensor.
   */
  forward(z) {
    function _relu(z2) {
      if (typeof z2[0] === "number") {
        return z2.map((el) => {
          if (el > 0) {
            return 1;
          } else {
            return 1e-3;
          }
        });
      } else if (typeof z2[0] === "object") {
        return z2.map((el) => _relu(el));
      } else throw Error("In ReLU, provided Tensor is not homogenous.");
    }
    const mask = tensor(_relu(z._data));
    z = z.mul(mask);
    return z;
  }
}
class Softmax extends Module {
  /**
   * Softmax nonlinearity class. Returns distribution of values (sum=1).
   */
  constructor() {
    super();
  }
  /**
   * Performs forward pass through Softmax nonlinearity.
   * @param {Tensor} z - input Tensor.
   * @param {number} dim - dimension across which to apply Softmax.
   * @returns {Tensor} new Tensor.
   */
  forward(z, dim = -1) {
    z = exp(z);
    const out = z.div(z.sum(dim, true));
    return out;
  }
}
class Dropout extends Module {
  p;
  /**
   * Dropout class, added usually after other layers, to drop values to zero with given probability
   *
   * @param {number} drop_prob - probability to drop each value in input.
   */
  constructor(drop_prob) {
    super();
    this.p = drop_prob;
    this.mode = "train";
  }
  /**
   * Performs forward pass through Dropout layer. Sets random values to zero (this.p % of the total).
   * @param {Tensor} z - input Tensor.
   * @returns {Tensor} new Tensor.
   */
  forward(z) {
    if (this.mode == "eval") {
      return z;
    }
    const mask = rand(z.shape);
    let a = z.masked_fill(
      mask,
      (el) => {
        return el < this.p;
      },
      0
    );
    a = a.div(1 - this.p);
    return a;
  }
}
class LayerNorm extends Module {
  gamma;
  beta;
  /**
   * Layer Norm class, added usually after other layers to normalize across all of the output.
   *
   * @param {number} n_embed - size of the last dimention of the input.
   */
  constructor(n_embed) {
    super();
    this.gamma = ones([n_embed], true);
    this.beta = zeros([n_embed], true);
  }
  forward(x) {
    const var_x = x.variance(-1, true);
    const norm_x = x.sub(x.mean(-1, true)).div(sqrt(var_x));
    const z = mul(norm_x, this.gamma).add(this.beta);
    return z;
  }
}
class CrossEntropyLoss extends Module {
  /**
   * Cross Entropy Loss class, returns the loss given the output and the expected indexes.
   */
  constructor() {
    super();
  }
  /**
   * Performs forward pass through CrossEntropyLoss, returns loss.
   * @param {Tensor} z - Output from the last layer of the network. Must have shape like (*Batch dimentions, Number of possible classes).
   * @param {object} y - Correct indexes expected from the model.
   * @returns {object} Negative-log-likelihood loss of the model output.
   */
  forward(z, y) {
    let zDims = z.shape;
    const D = zDims.slice(zDims.length - 1, zDims.length)[0];
    zDims = zDims.slice(0, zDims.length - 1);
    const B = zDims.reduce((a, b) => a * b, 1);
    z = z.reshape([B, D]);
    const logitsExp = exp(z);
    const logitsSum = logitsExp.sum(1, true);
    const logits = logitsExp.div(logitsSum);
    const y_array = _reshape(y.data, [B]);
    const at_logits = logits.at([...Array(B).keys()], y_array);
    const log_losses = log(at_logits);
    let loss = log_losses.sum(-1).neg();
    loss = loss.div(B);
    return loss;
  }
}
class MSELoss extends Module {
  /**
   * Constructor.
   */
  constructor() {
    super();
  }
  /**
   * Performs forward pass through MSELoss, returns loss.
   * @param {Tensor} z - Output from the last layer of the network.
   * @param {object} y - Correct outputs expected from the model.
   * @returns {object} Mean Squared Error loss of the model output.
   */
  forward(z, y) {
    let zDims = z.shape;
    const D = zDims.slice(zDims.length - 1, zDims.length)[0];
    zDims = zDims.slice(0, zDims.length - 1);
    const B = zDims.reduce((a, b) => a * b, 1);
    z = z.reshape([B, D]);
    y = y.reshape([B, D]);
    const S = z.sub(y);
    const P = S.pow(2);
    const Su = P.sum();
    let loss = Su.mean();
    loss = loss.div(B);
    return loss;
  }
}
function save(model, file) {
  function recursiveReplacer(obj) {
    let result = {};
    for (var x in obj) {
      if (x !== "forwardKernel" && x !== "backwardKernelA" && x !== "backwardKernelB" && x !== "gpu") {
        if (typeof obj[x] === "object" && !Array.isArray(obj[x])) {
          result[x] = recursiveReplacer(obj[x]);
        } else {
          result[x] = obj[x];
        }
      } else {
        result[x] = null;
      }
    }
    return result;
  }
  const replaced = recursiveReplacer(model);
  return replaced;
}
function load(model, file) {
  const loadedData = fs.readFileSync(file);
  let loadedModel = JSON.parse(loadedData.toString());
  loadParameters(loadedModel, model);
  return model;
}
function loadParameters(source, target) {
  for (const [key, value] of target.entries()) {
    if (value instanceof Module) {
      loadParameters(source[key], target[key]);
    } else if (value instanceof Parameter || value instanceof Tensor) {
      target[key]._data = source[key]._data;
      target[key].m = source[key].m;
      target[key].v = source[key].v;
    }
  }
}

class Adam {
  // Declare Adam's types:
  params;
  lr;
  reg;
  b1;
  b2;
  eps;
  /**
   * Adam optimizer class.
   * @param {(Parameter | Tensor)[]} params - List of all Parameter or Tensor (with requires_grad = True) to be optimized by Adam. "params" is usually set to nn.Module.parameters(), which automatically returns all parameters in a list form.
   * @param {number} lr - Scalar multiplying each learning step, controls speed of learning.
   * @param {number} reg - Scalar controling strength l2 regularization.
   * @param {(number)[]} betas - Two scalar floats controling how slowly the optimizer changes the "m" and "v" attributes.
   * @param {number} eps - Scalar added to denominator to stop it from ever going to zero.
   */
  constructor(params, lr = 1e-3, reg = 0, betas = [0.9, 0.99], eps = 1e-9) {
    this.params = params;
    this.lr = lr;
    this.reg = reg;
    this.b1 = betas[0];
    this.b2 = betas[1];
    this.eps = eps;
    this.reg = reg;
    for (let i = 0; i < this.params.length; i++) {
      this.params[i].m = zeros(this.params[i].shape);
      this.params[i].v = zeros(this.params[i].shape);
    }
  }
  /**
   * Updates all parameters in this.params with their gradients.
   */
  step() {
    for (let i = 0; i < this.params.length; i++) {
      this.params[i].m = this.params[i].m?.mul(this.b1).add(this.params[i]._grad?.mul(1 - this.b1));
      this.params[i].v = this.params[i].v?.mul(this.b2).add(this.params[i]._grad?.pow(2).mul(1 - this.b2));
      const update_tensor = this.params[i].m?.mul(this.lr).div(this.params[i].v?.sqrt().add(this.eps)).neg();
      const regularization_tensor = this.params[i].mul(this.reg * this.lr).neg();
      this.params[i]._data = this.params[i].add(
        update_tensor?.add(regularization_tensor)
      )._data;
    }
  }
  /**
   * Sets all the gradients of self.params to zero.
   */
  zero_grad() {
    for (let i = 0; i < this.params.length; i++) {
      this.params[i].zero_grad();
    }
  }
}

function tanh(x) {
  function _tanh(x2) {
    if (typeof x2[0] === "number") {
      return x2.map((el) => {
        const exp_x = Math.exp(el);
        const exp_neg_x = Math.exp(-el);
        return (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
      });
    } else if (typeof x2[0] === "object") {
      return x2.map((el) => _tanh(el));
    } else {
      throw Error("In tanh, provided Tensor is not homogenous.");
    }
  }
  const tensor = new Tensor(_tanh(x._data), x.requires_grad, x.device);
  return tensor;
}
function _calculate_fan_in_and_fan_out(shape) {
  if (shape.length < 2) {
    throw new Error("Tensor must have at least 2 dimensions for fan in/out calculation");
  }
  let fanIn;
  let fanOut;
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
function xavier_normal_(tensor, gain = 1) {
  const randomNormal = (mean = 0, std2 = 1) => {
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + z0 * std2;
  };
  const fillTensorData = (data, std2) => {
    if (Array.isArray(data)) {
      return data.map((item) => fillTensorData(item, std2));
    } else {
      return randomNormal(0, std2);
    }
  };
  const { fanIn, fanOut } = _calculate_fan_in_and_fan_out(tensor.shape);
  const std = gain * Math.sqrt(2 / (fanIn + fanOut));
  const newData = fillTensorData(tensor.data, std);
  const newTensor = new Tensor(newData, tensor.requires_grad, tensor.device);
  return newTensor;
}
function uniform_(tensor, a = 0, b = 1) {
  const randomUniform = () => {
    return a + Math.random() * (b - a);
  };
  const createNewData = (data) => {
    if (Array.isArray(data)) {
      return data.map((item) => createNewData(item));
    } else {
      return randomUniform();
    }
  };
  const newData = createNewData(tensor.data);
  const newTensor = new Tensor(newData, tensor.requires_grad, tensor.device);
  return newTensor;
}
function kaiming_uniform_(tensor, a = Math.sqrt(5), mode = "fan_in", nonlinearity = "leaky_relu") {
  let gain;
  if (nonlinearity === "leaky_relu") {
    gain = Math.sqrt(2 / (1 + a * a));
  } else if (nonlinearity === "relu") {
    gain = Math.sqrt(2);
  } else if (nonlinearity === "tanh") {
    gain = 5 / 3;
  } else {
    gain = 1;
  }
  const { fanIn, fanOut } = _calculate_fan_in_and_fan_out(tensor.shape);
  const fan = mode === "fan_in" ? fanIn : fanOut;
  const bound = gain * Math.sqrt(3 / fan);
  return uniform_(tensor, -bound, bound);
}
function zeros_(shape) {
  if (shape.length === 1) {
    return Array(shape[0]).fill(0);
  }
  return Array(shape[0]).fill(0).map(
    () => zeros_(shape.slice(1))
  );
}

class RecurrentLayer extends Module {
  W_ih;
  // Input-to-hidden weights
  W_hh;
  // Hidden-to-hidden weights
  b_ih;
  // Input-to-hidden bias
  b_hh;
  // Hidden-to-hidden bias
  constructor(input_size, hidden_size, nonlinearity = "tanh") {
    super();
    this.input_size = input_size;
    this.hidden_size = hidden_size;
    this.nonlinearity = nonlinearity;
    this.W_ih = new Parameter(zeros_([hidden_size, input_size]));
    this.W_hh = new Parameter(zeros_([hidden_size, hidden_size]));
    this.b_ih = new Parameter(zeros_([hidden_size]));
    this.b_hh = new Parameter(zeros_([hidden_size]));
  }
  reset_parameters() {
    this.W_ih = kaiming_uniform_(this.W_ih, Math.sqrt(5));
    this.W_hh = kaiming_uniform_(this.W_hh, Math.sqrt(5));
    const { fanIn } = _calculate_fan_in_and_fan_out([this.hidden_size, this.input_size]);
    const bound = fanIn > 0 ? 1 / Math.sqrt(fanIn) : 0;
    this.b_ih = uniform_(this.b_ih, -bound, bound);
    this.b_hh = uniform_(this.b_hh, -bound, bound);
  }
  linear(x_t, weight, bias) {
    const output = x_t.matmul(transpose(weight, 0, 1));
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
  forward(x, h_0) {
    const shape = x.shape;
    const batch_size = shape[0];
    const seq_len = shape[1];
    let h_t;
    if (!h_0) {
      h_t = zeros([batch_size, this.hidden_size], x.requires_grad, x.device);
    } else {
      h_t = h_0;
    }
    const output = zeros([batch_size, seq_len, this.hidden_size], x.requires_grad, x.device);
    for (let t = 0; t < seq_len; t++) {
      const x_t = x.slice([null, t, null]);
      const input_projection = this.linear(x_t, this.W_ih, this.b_ih);
      const hidden_projection = this.linear(h_t, this.W_hh, this.b_hh);
      const combined = input_projection.add(hidden_projection);
      h_t = tanh(combined);
      output.setSlice([null, t, null], h_t);
    }
    return [output, h_t];
  }
}
class SimpleRNN extends Module {
  embed;
  rnn;
  dense;
  constructor(vocab_size = 27, embedding_dim = 256, hidden_layers = 320) {
    super();
    this.embed = new Embedding(vocab_size, embedding_dim);
    this.rnn = new RecurrentLayer(embedding_dim, hidden_layers);
    this.dense = new Linear(hidden_layers, vocab_size);
    this.dense.W = xavier_normal_(this.dense.W);
    this.embed.E = xavier_normal_(this.embed.E);
  }
  forward(x) {
    const embedded = this.embed.forward(x);
    let [out, _] = this.rnn.forward(embedded);
    out = this.dense.forward(out.slice([null, out.shape[1] - 1, null]));
    return out;
  }
}
class Transformer extends Module {
  constructor(vocab_size = 27, hidden_size = 64, n_timesteps = 16, n_heads = 4, dropout_p = 0.2, device = "cpu") {
    super();
    this.embed = new Embedding(vocab_size, hidden_size);
    this.pos_embed = new PositionalEmbedding(n_timesteps, hidden_size);
    this.b1 = new Block(hidden_size, hidden_size, n_heads, n_timesteps, dropout_p, device);
    this.b2 = new Block(hidden_size, hidden_size, n_heads, n_timesteps, dropout_p, device);
    this.ln = new LayerNorm(hidden_size);
    this.linear = new Linear(hidden_size, vocab_size, device);
  }
  forward(x) {
    let z;
    z = add(this.embed.forward(x), this.pos_embed.forward(x));
    z = this.b1.forward(z);
    z = this.b2.forward(z);
    z = this.ln.forward(z);
    z = this.linear.forward(z);
    return z;
  }
}

const nn = {
  Module,
  Linear,
  MultiHeadSelfAttention,
  FullyConnected,
  Block,
  Embedding,
  PositionalEmbedding,
  ReLU,
  Softmax,
  Dropout,
  LayerNorm,
  CrossEntropyLoss,
  MSELoss
};
const optim = { Adam };
const models = { SimpleRNN, Transformer };
const torch = {
  // Add methods from tensor.js (these methods are accessed with "torch."):
  Tensor,
  Parameter,
  add,
  neg,
  mul,
  div,
  matmul,
  exp,
  log,
  sqrt,
  pow,
  mean,
  masked_fill,
  variance,
  at,
  reshape,
  _reshape,
  transpose,
  tensor,
  randint,
  randn,
  rand,
  tril,
  ones,
  zeros,
  argmax,
  broadcast,
  save,
  load,
  // Add submodules:
  models,
  nn,
  optim,
  getShape
};

export { nn, torch };
