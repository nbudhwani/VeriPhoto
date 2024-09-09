chrome.runtime.onInstalled.addListener(() => {
  chrome.tabs.query({}, (tabs) => {
    tabs.forEach((tab) => {
      // Ensure `tab` is valid and has a `url` property
      if (tab && typeof tab.url === 'string' && !tab.url.startsWith('chrome://')) {
        // Inject ort.js first
        chrome.scripting.executeScript({
          target: { tabId: tab.id },
          files: ['onnxruntime-web/ort.js'],  // Inject ort.js
        }, () => {
          // Then inject content.js
          chrome.scripting.executeScript({
            target: { tabId: tab.id },
            files: ['content.js'],  // Inject content.js after ort.js
          }, (result) => {
            console.log('Injected content.js after ort.js', result);
          });
        });
      } else {
        console.log(`Skipping injection for tab:`, tab);  // Log skipped tabs
      }
    });
  });
});