// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.CommandLine;
using FaceAiSharp.Validation;

var rc = new RootCommand("FaceAiSharp validation tools");

var db = new Option<FileInfo>(
    name: "--db",
    description: "File to use as db to store results and for continuation",
    getDefaultValue: () => new FileInfo("faceaisharp-validation.litedb"));
rc.AddGlobalOption(db);

var dbEmbeddingCollectionName = new Option<string>(
    name: "---db-embedding-collection-name",
    getDefaultValue: () => "ArcfaceEmbeddings");

var dataset = new Option<DirectoryInfo>(
    name: "--dataset",
    getDefaultValue: () => new DirectoryInfo(@"C:\Users\georg\Downloads\lfw\lfw"));

var pairsFile = new Option<FileInfo>(
    name: "--pairs-file",
    getDefaultValue: () => new FileInfo(@"C:\Users\georg\Downloads\lfw\pairs.txt"));

var arcfaceModel = new Option<FileInfo>(
    name: "--arcface-model",
    getDefaultValue: () => new FileInfo(@"C:\Users\georg\facePics\arcfaceresnet100-8\resnet100\resnet100.onnx"));

var scrfdModel = new Option<FileInfo>(
    name: "--scrfd-model",
    getDefaultValue: () => new FileInfo(@"C:\Users\georg\OneDrive\Dokumente\BlazorFace\ScrfdOnnx\scrfd_2.5g_bnkps.onnx"));

var eyeStateModel = new Option<FileInfo>(
    name: "--eyestate-model",
    getDefaultValue: () => new FileInfo(@"C:\Users\georg\OneDrive\Dokumente\BlazorFace\openvino_open-closed-eye-0001\open_closed_eye.onnx"));

var threshold = new Option<float>(
    name: "--threshold",
    getDefaultValue: () => 0.29f);

var binJpegs = new Option<DirectoryInfo>(
    name: "--bin-jpegs");
binJpegs.IsRequired = true;

var preprocMode = new Option<GenerateEmbeddings.PreprocessingMode>(
    name: "--prprocessing-mode",
    getDefaultValue: () => GenerateEmbeddings.PreprocessingMode.AffineTransform);

var generateEmbeddings = new Command("generate-embeddings") { dataset, arcfaceModel, scrfdModel, dbEmbeddingCollectionName, pairsFile, preprocMode };

var calcAllDistances = new Command("calc-all-distances") { dbEmbeddingCollectionName, threshold };

var calcPairsDistances = new Command("calc-pairs-distances") { dbEmbeddingCollectionName, threshold, pairsFile };

var countClosedEyes = new Command("count-closed-eyes") { dataset, scrfdModel, eyeStateModel };

var renameModelzooBinJpegs = new Command("rename-modelzoo-bin-jpegs") { binJpegs, pairsFile };

#pragma warning disable SA1116 // Split parameters should start on line after declaration
#pragma warning disable SA1117 // Parameters should be on same line or separate lines

generateEmbeddings.SetHandler(async (dataset, db, arcfaceModel, scrfdModel, dbEmbeddingCollectionName, pairsFile, preprocMode) =>
{
    using var cmd = new GenerateEmbeddings(dataset, db, arcfaceModel, scrfdModel, dbEmbeddingCollectionName, pairsFile, preprocMode);
    await cmd.Invoke();
}, dataset, db, arcfaceModel, scrfdModel, dbEmbeddingCollectionName, pairsFile, preprocMode);
rc.AddCommand(generateEmbeddings);

calcAllDistances.SetHandler((db, dbEmbeddingCollectionName, threshold) =>
{
    var calc = new CalculateAllDistances(db, dbEmbeddingCollectionName, threshold);
    calc.Invoke();
}, db, dbEmbeddingCollectionName, threshold);
rc.AddCommand(calcAllDistances);

calcPairsDistances.SetHandler((db, dbEmbeddingCollectionName, threshold, pairsFile) =>
{
    var calc = new CalculatePairsDistances(db, dbEmbeddingCollectionName, threshold, pairsFile);
    calc.Invoke();
}, db, dbEmbeddingCollectionName, threshold, pairsFile);
rc.AddCommand(calcPairsDistances);

countClosedEyes.SetHandler((dataset, db, scrfdModel, eyeStateModel) =>
{
    var cnt = new CountClosedEyes(dataset, db, scrfdModel, eyeStateModel);
    cnt.Invoke();
}, dataset, db, scrfdModel, eyeStateModel);
rc.AddCommand(countClosedEyes);

renameModelzooBinJpegs.SetHandler((binJpegs, pairsFile) =>
{
    var ren = new RenameModelzooBinJpegs(binJpegs, pairsFile);
    ren.Invoke();
}, binJpegs, pairsFile);
rc.AddCommand(renameModelzooBinJpegs);

return await rc.InvokeAsync(args);
