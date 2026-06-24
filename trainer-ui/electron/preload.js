const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('dopsdElectron', {
  productName: 'T8 D-OPSD Tranier',
});
