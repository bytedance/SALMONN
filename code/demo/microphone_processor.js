// Capture on the audio thread; network and page updates never gate sampling.
class MicrophoneProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.buffer = new Float32Array(Math.round(sampleRate * 0.4));
        this.length = 0;
        this.active = true;
        this.port.onmessage = ({data}) => {
            if (data === "stop") {
                this.active = false;
                this.flush();
                this.port.postMessage({type: "stopped"});
            }
        };
    }
    flush() {
        if (!this.length) return;
        const chunk = this.buffer.slice(0, this.length);
        this.port.postMessage(chunk.buffer, [chunk.buffer]);
        this.length = 0;
    }
    process(inputs) {
        const channels = inputs[0];
        if (this.active && channels.length) {
            for (let frame = 0; frame < channels[0].length; frame++) {
                let value = 0;
                for (const channel of channels) value += channel[frame];
                this.buffer[this.length++] = value / channels.length;
                if (this.length === this.buffer.length) this.flush();
            }
        }
        // The output remains silent, avoiding feedback into the microphone.
        return true;
    }
}
registerProcessor("meco-microphone", MicrophoneProcessor);
