import { statSync, readFileSync } from "node:fs";
import { join } from "node:path";
import ort from "onnxruntime-node";

const root = process.cwd();
const files = {
  segmentation: join(root, "models/yolo26s-manga-seg.onnx"),
  detection: join(root, "models/ppocrv6-det.onnx"),
  recognition: join(root, "models/ppocrv6-rec.onnx"),
  dictionary: join(root, "models/ppocrv6_dict.txt"),
};

for (const [name, path] of Object.entries(files)) {
  const size = statSync(path).size;
  if (size < 1024) throw new Error(`${name} asset is unexpectedly small: ${path} (${size} bytes)`);
}

const dictionary = readFileSync(files.dictionary, "utf8")
  .split("\n")
  .map((line) => line.replace(/\r$/, ""));
if (dictionary.at(-1) === "") dictionary.pop();
if (dictionary.length !== 18708) {
  throw new Error(`PP-OCRv6 dictionary has ${dictionary.length} entries; expected 18708.`);
}

const segmentation = await ort.InferenceSession.create(files.segmentation);
const detection = await ort.InferenceSession.create(files.detection);
const recognition = await ort.InferenceSession.create(files.recognition);

if (segmentation.outputNames.length < 2) {
  throw new Error("YOLO26s segmentation model must expose detections and mask prototypes.");
}
if (detection.outputNames.length !== 1) {
  throw new Error("PP-OCRv6 detector must expose exactly one probability-map output.");
}
if (recognition.outputNames.length !== 1) {
  throw new Error("PP-OCRv6 recognizer must expose exactly one CTC output.");
}

const recInput = new ort.Tensor("float32", new Float32Array(3 * 48 * 320), [1, 3, 48, 320]);
const recResult = await recognition.run({ [recognition.inputNames[0]]: recInput });
const recDimensions = recResult[recognition.outputNames[0]].dims.map(Number);
const recClasses = recDimensions.at(-1);
if (recClasses !== dictionary.length + 2) {
  throw new Error(
    `PP-OCRv6 recognizer exposes ${recClasses} classes; expected ${dictionary.length + 2}.`
  );
}

console.log(
  JSON.stringify(
    {
      files: Object.fromEntries(
        Object.entries(files).map(([name, path]) => [name, { path, bytes: statSync(path).size }])
      ),
      sessions: {
        segmentation: {
          inputs: segmentation.inputNames,
          outputs: segmentation.outputNames,
        },
        detection: { inputs: detection.inputNames, outputs: detection.outputNames },
        recognition: {
          inputs: recognition.inputNames,
          outputs: recognition.outputNames,
          classes: recClasses,
        },
      },
    },
    null,
    2
  )
);
