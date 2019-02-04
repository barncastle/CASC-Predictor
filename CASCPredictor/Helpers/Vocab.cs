using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CASCPredictor.Helpers
{
    internal class Vocab
    {
        private string[] filenames;
        private char[] samplePrefixes;
        private readonly Dictionary<char, int> charToIndex;
        private readonly Dictionary<int, char> indexToChar;

        public Vocab(Options options)
        {
            charToIndex = new Dictionary<char, int>();
            indexToChar = new Dictionary<int, char>();

            if (!File.Exists(options.DataPath))
                throw new ArgumentException("Invalid training data path");

            TryLoadData(options.DataPath);
        }


        /// <summary>
        /// Returns the filename at the provided index
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public string this[int i] => filenames[i];
        /// <summary>
        /// Returns the count of filenames
        /// </summary>
        public int Count { get; private set; }
        /// <summary>
        /// Returns the count of unique characters
        /// </summary>
        public int CharCount { get; private set; }


        /// <summary>
        /// Encodes a character as an index
        /// </summary>
        /// <param name="c"></param>
        /// <returns></returns>
        public int Encode(char c) => charToIndex[c];

        /// <summary>
        /// Decodes an index as a character
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        public char Decode(int i) => indexToChar[i];

        /// <summary>
        /// Returns a random char based on the first characters in the training data
        /// </summary>
        public char GetRandomSamplePrefix => samplePrefixes.PickRandom();

        private void TryLoadData(string filename)
        {
            // load filenames and shuffle them to vary training
            filenames = File.ReadAllLines(filename).Shuffle();

            if (filenames.Length == 0)
                throw new InvalidDataException("Training data is empty");
            if (filenames.All(x => string.IsNullOrWhiteSpace(x)))
                throw new InvalidDataException("Training data is empty");

            // create a list of filename starting chars used for random sampling
            samplePrefixes = filenames.Select(x => x[0]).Distinct().ToArray();

            // build a char-index map of unique characters
            // a space is required for the one hot
            var characters = filenames.SelectMany(x => x).Distinct().ToList();
            if (!characters.Contains(' '))
                characters.Add(' ');
            characters.Sort();

            for (int i = 0; i < characters.Count; i++)
            {
                charToIndex[characters[i]] = i;
                indexToChar[i] = characters[i];
            }

            // vars
            Count = filenames.Length;
            CharCount = characters.Count;
        }
    }
}
