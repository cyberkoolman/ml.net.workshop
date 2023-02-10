using Microsoft.ML;

List<string> contents = new List<string>() {
  "What is the periodic table and how is it organized?",
  "What should be the proper order of calculation? A) multiplication B) addition C) subtraction D) paranthesis",
  "What is not the unit of force? A) F B) N C) lbf D) kg"
};

List<string> modelNames = new List<string>() {
  "subjects.zip",
  "subjects-10k.zip"
};

MLContext mlContext = new MLContext();
foreach (var modelName in modelNames)
{
  var model = mlContext.Model.Load(modelName, out var _);

  // Generate a prediction engine
  PredictionEngine<ModelInput, ModelOutput> engine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

  Console.WriteLine($"Inferencing Model: {modelName}");

  foreach (var content in contents)
  {
    Console.WriteLine($"Content: {content}");
    ModelInput sampleData = new() { Content = content };
    ModelOutput result = engine.Predict(sampleData);

    Console.WriteLine($"Class: {result.PredictedLabel}");  
    Console.WriteLine("----------------------------");
  }

  Console.WriteLine("----------------------------------------------------------------------------");
}
