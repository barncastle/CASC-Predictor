using System;
using CASCPredictor.Helpers;
using CommandLine;

namespace CASCPredictor
{
    class Program
    {
        static void Main(string[] args)
        {
            var parser = Parser.Default.ParseArguments<Options>(args);

            if (parser.Tag == ParserResultType.Parsed)
                parser.WithParsed(Run);

            Console.ReadLine();
        }

        private static void Run(Options options)
        {
            FileValidator.Load();
            var predictor = new FilePredictor(options);

            switch (options.Mode)
            {
                case Mode.Train:
                    {
                        predictor.Train();
                        Console.WriteLine($"Finished training {options.Epochs} epochs");
                        break;
                    }
                case Mode.Sample:
                    {
                        predictor.Sample().ForEach(x => Console.WriteLine(x));
                        Console.WriteLine($"Finished generating {options.SampleCount} samples");
                        break;
                    }
                default:
                    {
                        Console.WriteLine("Invalid Mode. Use 0 for Sampling or 1 for Training");
                        break;
                    }
            }

            FileValidator.Sync();
            Console.ReadLine();
        }
    }
}
