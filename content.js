console.log("Content script loaded.");

// Ensure `ort` is defined since `ort.js` is now statically loaded
if (typeof ort === 'undefined') {
  console.error("ONNX Runtime Web is not loaded.");
} else {
  console.log("ONNX Runtime Web is available globally:", ort);

  // Set the log level for ONNX Runtime to verbose for detailed debugging information
  ort.env.logLevel = 'verbose';  // Options: 'verbose', 'info', 'warning', 'error'

  // Set up the path for WebAssembly files
  ort.env.wasm.wasmPaths = chrome.runtime.getURL('onnxruntime-web/');

  const wasmLoaderUrl = chrome.runtime.getURL('onnxruntime-web/ort-wasm-simd-threaded.mjs');
  import(wasmLoaderUrl).then(() => {
    console.log("WASM module loaded successfully.");

    // Load ONNX model and run inference
    loadModelAndRunInference();
  }).catch((error) => {
    console.error("Failed to load WASM module:", error);
  });
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// Function to load the model and process images
async function loadModelAndRunInference() {
  try {
    // Load the ONNX model using the runtime URL
    const session = await ort.InferenceSession.create(chrome.runtime.getURL('model.onnx'));
    console.log('ONNX model loaded successfully!');

    // Process images directly
    document.querySelectorAll('img').forEach((imgElement) => {
      processImage(imgElement, session);
    });

  } catch (error) {
    console.error("Failed to load ONNX model:", error);
  }
}

// Function to process a single image
function processImage(imgElement, session) {
  console.log("Processing image:", imgElement.src);

  const img = new Image();
  img.crossOrigin = 'Anonymous';
  img.src = imgElement.src;

  img.onload = async () => {
    const imageData = getImageData(img);

    if (imageData) {
      console.log("Image data prepared for inference.");
      const isDeepfake = await runInference(session, imageData);
      console.log("Inference result:", isDeepfake);

      if (isDeepfake) {
        imgElement.style.border = "5px solid red";
        console.log("Deepfake detected for this image.");
      } else {
        console.log("This image is not a deepfake.");
      }
    } else {
      console.error("Failed to get image data for:", imgElement.src);
    }
  };

  img.onerror = (error) => {
    console.error("Error loading image:", error);
  };
}

// Function to get image data from the DOM
function getImageData(imgElement) {
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');

  imgElement.crossOrigin = 'Anonymous';

  ctx.drawImage(imgElement, 0, 0, 224, 224);

  try {
    const imageData = ctx.getImageData(0, 0, 224, 224);
    console.log("Successfully got image data for:", imgElement.src);
    console.log("Raw Image Data (first 10 values):", imageData.data.slice(0, 10));
    return imageData;
  } catch (e) {
    console.error('Error accessing canvas data for:', imgElement.src, e);
    return null;
  }
}

// Function to run inference using the ONNX model
async function runInference(session, imageData) {
  try {
    const inputTensor = preprocessImageData(imageData);
    console.log("Input Tensor Shape:", inputTensor.dims);
    console.log("Input Tensor Data (first 50 values):", inputTensor.data.slice(0, 50));

    const feeds = { input: inputTensor };
    const output = await session.run(feeds);
    console.log("Inference executed successfully!");

    const outputName = session.outputNames[0];
    const logit = output[outputName].data[0];
    console.log("Model output logit of being a deepfake:", logit);

    const probability = sigmoid(logit);
    console.log("Model output probability of being a deepfake (logit):", logit);
    const probabilityPercentage = (probability * 100).toFixed(2);
    console.log("Model output probability of being a deepfake (%):", probabilityPercentage + '%');

    return probability > 0.5;
  } catch (error) {
    console.error("Error during inference:", error);
    return null;
  }
}

// Function to preprocess image data for ONNX model
function preprocessImageData(imageData) {
  const width = imageData.width;
  const height = imageData.height;
  const data = imageData.data;

  const resizedData = resizeImageData(data, width, height, 224, 224);
  const floatData = normalizeImageData(resizedData);

  return new ort.Tensor('float32', floatData, [1, 3, 224, 224]);
}

// Function to resize image data
function resizeImageData(data, originalWidth, originalHeight, targetWidth, targetHeight) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = targetWidth;
  canvas.height = targetHeight;

  const imgData = new ImageData(new Uint8ClampedArray(data), originalWidth, originalHeight);
  ctx.putImageData(imgData, 0, 0);
  ctx.drawImage(canvas, 0, 0, originalWidth, originalHeight, 0, 0, targetWidth, targetHeight);

  return ctx.getImageData(0, 0, targetWidth, targetHeight).data;
}

// Function to normalize image data
function normalizeImageData(data) {
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const normalizedData = new Float32Array(data.length / 4 * 3);

  for (let i = 0; i < data.length / 4; i++) {
    const r = data[i * 4 + 0] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    normalizedData[i * 3 + 0] = (r - mean[0]) / std[0];
    normalizedData[i * 3 + 1] = (g - mean[1]) / std[1];
    normalizedData[i * 3 + 2] = (b - mean[2]) / std[2];
  }

  return normalizedData;
}