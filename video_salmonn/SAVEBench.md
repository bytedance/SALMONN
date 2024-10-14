# SAVE Benchmark
We have now uploaded the SAVE Bench [here](https://huggingface.co/datasets/tsinghua-ee/SAVEBench). 

## Details
Audio:
1. LibriSpeech test clean full set
2. Audio caps test full set

Video:
1. NExTQA: nextqa_test.json
ID provided in the "image" field

Image:
1. Flickr30k: flickr30k_captions.json
(this is the standard 1k test set). ID provided in the "image" field.
2. TextVQA: textvqa.json
ID provided in the "image" field
3. GQA: testdev_balanced_questions_with_images.json
ID provided in the "image" field

Audio-visual:
1. How2: how2_test.json
ID provided in "image". Format: <video_id>_<start_second>_<end_second>.mp4 or .wav.

2. Audio-Visual Sound Source Detection (AVSSD): testdata_formatted.json
ID provided in the "image" field. The first one is image and the second one is the corresponding audio.

3. Audio Visual Matching (AVM): audiovisualmatching_combined.json
ID provided in the "image" field as a list of two values. The first one is the image and the second one is the audio/speech
Whether it is from VGGSS or is from SpokenCOCO is indicated in the ID as well

4. Audio-visual question answering (AVQA) Ego4D-QA: ego4d_qa.json
"image_name" is given by: 14e96091-4011-4557-95b3-a195fb5c39d8__2.mp4
where "14e96091-4011-4557-95b3-a195fb5c39d8" is the video ID.
Duration is also provided.

5. Audio-visual question answering (AVQA) Presentation-QA: presentation_qa.json
Please download the dataset videos from: https://arxiv.org/pdf/2403.14168
The video ID and durations are provided in the file.
