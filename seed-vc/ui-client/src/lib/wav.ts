// Minimal WAV encoder for mono AudioBuffer -> ArrayBuffer (16-bit PCM)
export function audioBufferToWav(buffer: AudioBuffer): ArrayBuffer {
  const numChannels = buffer.numberOfChannels || 1;
  const sampleRate = buffer.sampleRate;
  const channelData = [] as Float32Array[];
  for (let c = 0; c < numChannels; c++) {
    channelData.push(buffer.getChannelData(c));
  }
  const length = channelData[0].length;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = length * blockAlign;
  const bufferSize = 44 + dataSize;
  const view = new DataView(new ArrayBuffer(bufferSize));

  let offset = 0;
  const writeString = (s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
    offset += s.length;
  };
  const writeUint32 = (v: number) => { view.setUint32(offset, v, true); offset += 4; };
  const writeUint16 = (v: number) => { view.setUint16(offset, v, true); offset += 2; };

  writeString('RIFF');
  writeUint32(bufferSize - 8);
  writeString('WAVE');
  writeString('fmt ');
  writeUint32(16); // PCM chunk size
  writeUint16(1); // PCM format
  writeUint16(numChannels);
  writeUint32(sampleRate);
  writeUint32(byteRate);
  writeUint16(blockAlign);
  writeUint16(bytesPerSample * 8);
  writeString('data');
  writeUint32(dataSize);

  // Interleave and clamp
  for (let i = 0; i < length; i++) {
    for (let c = 0; c < numChannels; c++) {
      let sample = channelData[c][i];
      sample = Math.max(-1, Math.min(1, sample));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }

  return view.buffer;
}
