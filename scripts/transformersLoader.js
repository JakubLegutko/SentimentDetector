(function () {
  const TRANSFORMERS_URL = chrome.runtime.getURL("vendor/transformers.min.js");
  let importPromise = null;

  async function loadTransformersRuntime() {
    if (window.transformers) {
      return window.transformers;
    }
    if (!importPromise) {
      importPromise = import(/* webpackIgnore: true */ TRANSFORMERS_URL).then((mod) => {
        window.transformers = mod;
        return mod;
      });
    }
    return importPromise;
  }

  window.loadTransformersRuntime = loadTransformersRuntime;
})();

