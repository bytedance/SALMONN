async (key) => {
    const base = new URL("./microphone/", window.location.href);
    const socketUrl = new URL(`socket/${key}`, base);
    socketUrl.protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(socketUrl);
    await new Promise((resolve, reject) => {
        socket.onopen = resolve;
        socket.onerror = () => reject(new Error("Cannot connect the microphone stream."));
    });
    let stream, context;
    try {
        context = new AudioContext();
        await context.audioWorklet.addModule(new URL("processor.js", base));
        stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1}, video: false});
        const source = context.createMediaStreamSource(stream);
        const processor = new AudioWorkletNode(context, "meco-microphone");
        const capture = {frames: 0, sampleRate: context.sampleRate, active: true, socket, stop: null};
        window.mecoMicrophone = capture;
        socket.send(JSON.stringify({sample_rate: context.sampleRate}));
        let resolveStopped, lastStatus = 0;
        const stopped = new Promise(resolve => { resolveStopped = resolve; });
        const status = text => {
            const element = document.querySelector("#microphone-capture-status .prose");
            if (element) element.textContent = text;
        };
        const closeCapture = async () => {
            capture.active = false;
            source.disconnect();
            processor.disconnect();
            stream.getTracks().forEach(track => track.stop());
            await context.close();
            resolveStopped();
        };
        capture.stop = async () => {
            if (capture.active) {
                capture.active = false;
                processor.port.postMessage("stop");
            }
            await stopped;
            return "Recording stopped. Finishing animation…";
        };
        processor.port.onmessage = ({data}) => {
            if (data instanceof ArrayBuffer) {
                capture.frames += data.byteLength / 4;
                socket.send(data);
                if (capture.frames - lastStatus >= capture.sampleRate * 0.8) {
                    lastStatus = capture.frames;
                    status(`Recording · ${(capture.frames / capture.sampleRate).toFixed(1)} s`);
                }
            } else if (data.type === "stopped") {
                // This follows every PCM packet, including the partial final packet.
                socket.send("stop");
            }
        };
        socket.onmessage = async ({data}) => {
            const message = JSON.parse(data);
            if (message.type === "limit") capture.stop();
            if (message.type === "finished") {
                capture.receivedFrames = message.source_frames;
                await closeCapture();
                socket.close();
            }
            if (message.type === "error") {
                status(message.message);
                await closeCapture();
                socket.close();
            }
        };
        socket.onclose = () => {
            if (context.state !== "closed") {
                status("Microphone connection closed. Recording stopped.");
                closeCapture();
            }
        };
        source.connect(processor);
        processor.connect(context.destination);
        await context.resume();
        return "Recording…";
    } catch (error) {
        stream?.getTracks().forEach(track => track.stop());
        if (context && context.state !== "closed") await context.close();
        socket.close();
        throw error;
    }
}
