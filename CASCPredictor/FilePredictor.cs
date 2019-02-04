using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using CASCPredictor.Functions;
using CASCPredictor.Helpers;
using CNTK;

namespace CASCPredictor
{
    using Arguments = Dictionary<Variable, Value>;

    public class FilePredictor
    {
        public const string ModelDirectory = "Models";

        private const int Layers = 2;
        private const int HiddenDimensions = 256;

        private readonly Options Options;
        private readonly DeviceDescriptor Device;
        private readonly Vocab Vocab;

        private Function Model;
        private IOPair<Variable> Inputs;
        private uint Revision;

        public FilePredictor(Options options, DeviceDescriptor device = null)
        {
            Options = options ?? throw new ArgumentNullException("No options provided");
            Device = device ?? DeviceDescriptor.UseDefaultDevice();
            Vocab = new Vocab(options);

            GetOrCreateModel();
        }


        #region Training

        public void Train()
        {
            Trainer trainer = BuildTrainer();

            // start the training
            Console.WriteLine($"Running {Options.Epochs} epochs with {Vocab.Count} batches per epoch");
            Console.WriteLine();

            for (int i = 0; i < Options.Epochs; i++)
            {
                DateTime startTime = DateTime.Now;
                IOPair<float[]> sequenceData;
                Arguments arguments = new Arguments();

                Console.WriteLine($"Running training on epoch {i + 1} of {Options.Epochs}");

                for (int j = 0; j < Vocab.Count; j++)
                {
                    sequenceData = GetSequenceData(j);

                    arguments[Inputs.Input] = Value.CreateSequence(Inputs.Input.Shape, sequenceData.Input, Device);
                    arguments[Inputs.Output] = Value.CreateSequence(Inputs.Output.Shape, sequenceData.Output, Device);

                    trainer.TrainMinibatch(arguments, false, Device);

                    // sample
                    if (j % Options.SampleFrequency == 0)
                    {
                        string sample = Sample(Options.SampleSize);
                        FileValidator.Validate(sample);
                        Console.WriteLine(sample);
                    }


                    // print current state
                    if (j % 100 == 0)
                    {
                        Console.WriteLine(string.Format("Epoch {0}: Batch [{1}-{2}] Cross Entropy = {3}, Evaluation = {4}",
                            (i + 1).ToString("000"),
                            (j + 1).ToString("000000"),
                            (j + 100).ToString("000000"),
                            trainer.PreviousMinibatchLossAverage().ToString("F6"),
                            trainer.PreviousMinibatchEvaluationAverage().ToString("F3")
                        ));
                    }
                }

                // log completion
                string epochTime = (DateTime.Now - startTime).ToString(@"hh\:mm\:ss\.fff");
                Console.WriteLine($"Finished epoch {i + 1}: {epochTime}");

                // save the model
                string modelFileName = Path.Combine(ModelDirectory, $"{Options.ModelPrefix}_epoch{++Revision}.dnn");
                Model.Save(modelFileName);
                Console.WriteLine($"Saved model to {modelFileName}");
            }
        }

        private Trainer BuildTrainer()
        {
            //  setup the cross entropy loss
            var crossEntropy = CNTKLib.CrossEntropyWithSoftmax(Model, Inputs.Output);
            var errorFunction = CNTKLib.ClassificationError(Model, Inputs.Output);

            //  create the trainer
            var learningRateSchedule = new TrainingParameterScheduleDouble(0.001, 1);
            var momentumSchedule = CNTKLib.MomentumAsTimeConstantSchedule(1100);
            var parameters = new AdditionalLearningOptions
            {
                gradientClippingThresholdPerSample = 5.0,
                gradientClippingWithTruncation = true
            };
            var learner = Learner.MomentumSGDLearner(Model.Parameters(), learningRateSchedule, momentumSchedule, true, parameters);

            return Trainer.CreateTrainer(Model, crossEntropy, errorFunction, new List<Learner>() { learner });
        }

        private IOPair<float[]> GetSequenceData(int index)
        {
            string filename = Vocab[index];
            int length = filename.Length;

            float[] input = new float[Vocab.CharCount * length];
            float[] output = new float[Vocab.CharCount * length];

            for (int i = 0; i < length; i++)
                input[(Vocab.CharCount * i) + Vocab.Encode(filename[i])] = 1;

            // copy the input sequence shifted by one
            // space fill the final output char
            Array.Copy(input, 1, output, 0, input.Length - 1);
            output[output.Length - Vocab.CharCount + Vocab.Encode(' ')] = 1;

            return new IOPair<float[]>(input, output);
        }

        #endregion

        #region Sampling

        public IEnumerable<string> Sample()
        {
            for (int i = 0; i < Options.SampleCount; i++)
            {
                string sample = Sample(Options.SampleSize, Options.SamplePrime);

                // the sample is filled to the samplesize with garbage so needs some minimal validation
                // ' ' is used as the one hot trains the model all filenames must end with it
                // these are then further filtered by the supplied prime
                string[] results = sample.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                foreach (string result in results)
                {
                    if(result.StartsWith(Options.SamplePrime))
                    {
                        FileValidator.Validate(result);
                        yield return result;
                    }
                }
            }
        }

        private string Sample(int targetLength, string prime = null)
        {
            Variable inputVariable = Model.Arguments[0];
            Variable outputVariable = Model.Output;

            var inputs = new Arguments();
            var outputs = new Arguments();

            // uses a random recommended char if no prime is provided
            if (string.IsNullOrEmpty(prime))
                prime = Vocab.GetRandomSamplePrefix.ToString();

            var sequence = new List<float>(Vocab.CharCount * targetLength);
            char[] result = new char[targetLength];

            // load the prefix into the sample
            int i = 0, charIndex = 0;
            for (; i < prime.Length; i++)
            {
                charIndex = Vocab.Encode(prime[i]);
                UpdateAndEvaluateModel();

                result[i] = prime[i];
            }

            // fill the remainder of the sample with suggested chars
            for (; i < targetLength; i++)
            {
                var outputData = outputs[outputVariable].GetDenseData<float>(outputVariable)[0] as List<float>;
                charIndex = GetSuggestedIndex(outputData, i);
                UpdateAndEvaluateModel();

                // add the suggest char to the result string
                result[i] = Vocab.Decode(charIndex);
            }

            void UpdateAndEvaluateModel()
            {
                // map the used char to the sequence
                // - using a List decreases the round trip times
                float[] input = new float[Vocab.CharCount];
                input[charIndex] = 1;
                sequence.AddRange(input);

                // re-evaluate the model
                inputs[inputVariable] = Value.CreateSequence(inputVariable.Shape, sequence, Device);
                outputs[outputVariable] = null;
                Model.Evaluate(inputs, outputs, Device);
            }

            return new string(result);
        }

        private int GetSuggestedIndex(List<float> outputData, int charIndex)
        {
            float probsSum = 0;
            float randomValue = (float)new Random().NextDouble();

            // get the probabilities for the specific char
            var probs = outputData.GetRange(Vocab.CharCount * (charIndex - 1), Vocab.CharCount);

            // normalise and sum
            for (int i = 0; i < probs.Count; i++)
                probsSum += probs[i] = (float)Math.Exp(probs[i]);

            for (int i = 0; i < probs.Count; i++)
            {
                // temperature: T < 1 = smoother; T=1.0 = same; T > 1 = more peaked
                // Math.Pow(probs[i] / probsSum, temperature) / probsSum
                // - shortcut pow as currently always 1.0
                if ((randomValue -= probs[i] / probsSum) < 0)
                    return i;
            }

            // fallback to the final char in the vocab
            return probs.Count - 1;
        }

        #endregion

        #region Helpers

        private void GetOrCreateModel()
        {
            Directory.CreateDirectory(ModelDirectory);

            // create the inputs
            Axis axis = new Axis("inputAxis");
            var features = Variable.InputVariable(new[] { Vocab.CharCount }, DataType.Float, "features", new List<Axis> { axis, Axis.DefaultBatchAxis() });
            var labels = Variable.InputVariable(new[] { Vocab.CharCount }, DataType.Float, "labels", new List<Axis> { axis, Axis.DefaultBatchAxis() });

            if (TryGetModel(out string filename))
            {
                // load the previous model and use it's features
                // the labels should be identical as before
                Model = Function.Load(filename, Device);
                Inputs = new IOPair<Variable>(Model.Arguments[0], labels);
                Console.WriteLine($"Loaded {Path.GetFileName(filename)}");
            }
            else
            {
                // create a new model from the features
                Model = features;
                for (int i = 0; i < Layers; i++)
                {
                    Model = Stabilizer.Build(Model, Device);
                    Model = LSTM.Build(Model, HiddenDimensions, Device);
                }

                Model = Dense.Build(Model, Vocab.CharCount, Device);
                Inputs = new IOPair<Variable>(features, labels);
            }
        }

        private bool TryGetModel(out string filename)
        {
            filename = null;
            Regex regex = new Regex(@"(\d+)\.dnn", RegexOptions.Compiled | RegexOptions.IgnoreCase);

            // find files with the model prefix
            var files = Directory.EnumerateFiles(ModelDirectory, $"{Options.ModelPrefix}_epoch*.dnn");
            foreach (var file in files)
            {
                // ignore improperly named files
                var match = regex.Match(file);
                if (!match.Success)
                    continue;

                // validate the revision and take the highest
                if (uint.TryParse(match.Groups[1].Value, out uint rev) && rev > Revision)
                {
                    Revision = rev;
                    filename = file;
                }
            }

            return Revision > 0;
        }

        #endregion
    }
}
