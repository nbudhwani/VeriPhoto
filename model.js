// Log a message to verify that the ONNX Runtime Web is available
console.log("ONNX Runtime Web imported successfully!", ort);  // 'ort' is available globally now

document.addEventListener('DOMContentLoaded', function () {
  // Set up the button to execute the content script
  document.getElementById('refresh').addEventListener('click', function () {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.scripting.executeScript({
        target: { tabId: tabs[0].id },
        files: ['content.js']
      });
    });
  });

  loadModelAndRunInference();
});

// Load the ONNX model and run inference on images
async function loadModelAndRunInference() {
  try {
    // Load the ONNX model
    const session = await ort.InferenceSession.create('model.onnx'); // Replace with your ONNX model file name
    console.log('ONNX model loaded successfully!');

    // Get images from the DOM and run inference
    document.querySelectorAll('img').forEach(async (img) => {
      const imageData = getImageData(img);
      console.log('It got to before the imageData break!');
      if (imageData) {
        console.log("Image data prepared for inference.");
        // Run inference on the image data
        const isDeepfake = await runInference(session, imageData);
        console.log("Inference result:", isDeepfake);

        // Highlight the image if it is a deepfake
        if (isDeepfake) {
          img.style.border = "5px solid red";  // Highlight detected deepfakes
          console.log("Deepfake detected for this image.");
        } else {
          console.log("This image is not a deepfake.");
        }
      }
    });
  } catch (error) {
    console.error("Failed to load ONNX model or run inference:", error);
  }
}

async function runInference(session, inputTensor) {
  try {
    const feeds = { input: inputTensor };
    const output = await session.run(feeds);

    const outputName = session.outputNames[0];
    const logits = output[outputName].data;

    const logit = logits[0];
    const realProbability = 1 / (1 + Math.exp(-logit)); // Sigmoid function
    const fakeProbability = 1 - realProbability;

    console.log("Model output logit:", logit);
    console.log("Probability of being real:", realProbability);
    console.log("Probability of being fake:", fakeProbability);

    // Determine if the image is a deepfake based on the fake probability
    return fakeProbability > 0.5;
  } catch (error) {
    console.error("Error during inference:", error);
    return null;
  }
}

function preprocessImageData(imageData) {
  const width = imageData.width;
  const height = imageData.height;
  const data = imageData.data;

  const floatData = new Float32Array(1 * 3 * height * width);

  // Loop over all pixels and arrange the data
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const r = data[idx * 4 + 0] / 255.0;
      const g = data[idx * 4 + 1] / 255.0;
      const b = data[idx * 4 + 2] / 255.0;

      // Assign to floatData in channel-first order
      floatData[0 * height * width + y * width + x] = r; // Red channel
      floatData[1 * height * width + y * width + x] = g; // Green channel
      floatData[2 * height * width + y * width + x] = b; // Blue channel
    }
  }

  return new ort.Tensor('float32', floatData, [1, 3, height, width]);
}

// Function to resize image data
function resizeImageData(data, originalWidth, originalHeight, targetWidth, targetHeight) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = targetWidth;
  canvas.height = targetHeight;

  const imageData = new ImageData(new Uint8ClampedArray(data), originalWidth, originalHeight);
  ctx.putImageData(imageData, 0, 0);

  ctx.drawImage(canvas, 0, 0, originalWidth, originalHeight, 0, 0, targetWidth, targetHeight);
  return ctx.getImageData(0, 0, targetWidth, targetHeight).data;
}

//Removed Mean and Standard Deviation Normalization: Since the model handles normalization internally, we only need to scale the pixel values to the [0, 1] range by dividing by 255.0.
//Converted to Float32Array: Ensures the data is in the correct format for the ONNX model.
//Excluded Alpha Channel: If your images have an alpha channel (RGBA), we only need the RGB channels.

function normalizeImageData(data) {
  const normalizedData = new Float32Array(data.length / 4 * 3);

  for (let i = 0; i < data.length / 4; i++) {
    const r = data[i * 4 + 0] / 255.0; // Red channel
    const g = data[i * 4 + 1] / 255.0; // Green channel
    const b = data[i * 4 + 2] / 255.0; // Blue channel

    normalizedData[i * 3 + 0] = r;
    normalizedData[i * 3 + 1] = g;
    normalizedData[i * 3 + 2] = b;
  }

  return normalizedData;
}


// Function to get image data from the DOM
function getImageData(imgElement) {
  const canvas = document.createElement('canvas');
  canvas.width = 224;  // Adjust size as needed
  canvas.height = 224;
  const ctx = canvas.getContext('2d');

  // Handle cross-origin images
  imgElement.crossOrigin = 'Anonymous';  // Set crossOrigin to 'Anonymous'

  // Draw the image on the canvas
  ctx.drawImage(imgElement, 0, 0, 224, 224);  // Resize and draw

  try {
    const imageData = ctx.getImageData(0, 0, 224, 224);
    return imageData;
  } catch (e) {
    console.error('Error accessing canvas data:', e);
    return null;
  }

}