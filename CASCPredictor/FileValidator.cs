using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using CASCPredictor.Helpers;

namespace CASCPredictor
{
    static class FileValidator
    {
        private const string FileName = "unk_listfile.txt";
        private const int MaxFileCount = 20000;

        private static readonly HashSet<ulong> Hashes;
        private static readonly Queue<string> Filenames;
        private static readonly Jenkins96 Jenkins96;

        static FileValidator()
        {
            Hashes = new HashSet<ulong>();
            Filenames = new Queue<string>();
            Jenkins96 = new Jenkins96();
        }

        
        public static void Load()
        {
            GetUnknownListfile();
            LoadHashes();
        }

        public static void Validate(string filename)
        {
            if (Hashes.Contains(Jenkins96.ComputeHash(filename)))
                Filenames.Enqueue(filename);
        }

        public static void Sync()
        {
            if (Filenames.Count == 0)
                return;

            Console.WriteLine($"Found {Filenames.Count} filenames:");
            Filenames.ForEach(x => Console.WriteLine(x));
            PostResults();
        }


        private static void GetUnknownListfile()
        {
            if (File.Exists(FileName) && (DateTime.Now - File.GetLastWriteTime(FileName)).TotalHours < 6)
                return;

            try
            {
                HttpWebRequest req = (HttpWebRequest)WebRequest.Create("https://bnet.marlam.in/listfile.php?unk=1&t=" + DateTime.Now.Ticks);
                req.UserAgent = "CASCPredictor/1.0 (+https://github.com/barncastle/CASC-Predictor)";

                using (WebResponse resp = req.GetResponse())
                using (FileStream fs = File.Create(FileName))
                    resp.GetResponseStream().CopyTo(fs);

                req.Abort();
            }
            catch
            {
                throw new Exception($"Unable to download unknown listfile");
            }
        }

        private static void LoadHashes()
        {
            var lines = File.ReadAllLines(FileName).Select(x => x.Trim());

            foreach (var line in lines)
            {
                if (ulong.TryParse(line, NumberStyles.HexNumber, null, out ulong dump))
                    Hashes.Add(dump);
                if (ulong.TryParse(line, out dump))
                    Hashes.Add(dump);
            }
        }

        private static void PostResults()
        {
            int pages = (Filenames.Count + MaxFileCount - 1) / MaxFileCount;
            for (int i = 0; i < pages; i++)
            {
                try
                {
                    string payload = string.Join("\r\n", Filenames.DequeueRange(MaxFileCount));
                    byte[] payload_data = Encoding.ASCII.GetBytes("files=" + payload);

                    HttpWebRequest req = (HttpWebRequest)WebRequest.Create("https://bnet.marlam.in/checkFiles.php");
                    req.Method = "POST";
                    req.UserAgent = "CASCPredictor/1.0 (+https://github.com/barncastle/CASC-Predictor)";
                    req.ContentType = "application/x-www-form-urlencoded";
                    req.ContentLength = payload_data.Length;

                    using (var stream = req.GetRequestStream())
                    {
                        stream.Write(payload_data);
                        req.GetResponse();
                    }

                    req.Abort();
                }
                catch
                {
                    Console.WriteLine("Unabled to post found filenames to `https://bnet.marlam.in/checkFiles.php`");
                }
            }
        }
    }
}
