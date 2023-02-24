using System.Text.Json;

const string vocabularyFile = "vocab.txt";
const string bertModelFile = "bertsquad-10.onnx";
const string contextFile = "context.txt";
const string questionFile = "question.txt";

var model = new Bert(vocabularyFile, bertModelFile);

var contextText = "";
var questionText = "";

if (args == null || args.Length == 0)
{
    contextText = File.ReadAllText(contextFile);
    questionText = File.ReadAllText(questionFile);
}
else 
{
    contextText = args[0];
    questionText = args[1];
}

var (tokens, probability) = model.Predict(context:contextText, question:questionText);

Console.WriteLine($"Question: {questionText}");
Console.WriteLine(JsonSerializer.Serialize(new
{
    Tokens = tokens,
    Probability = probability
}));