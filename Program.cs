using System;
using MWDLLDetectQKT;
using MLApp;
using MathWorks.MATLAB.NET.Arrays;
using System.Diagnostics;
using Emgu.CV.Structure;
using Emgu.CV;
using System.Collections;
using Emgu.CV.UI;
using System.IO;
using System.Collections.Generic;

namespace testDLL
{
    class Program
    {
        public static String[] GetFilesFrom(String searchFolder, String[] filters, bool isRecursive)
        {
            List<String> filesFound = new List<String>();
            var searchOption = isRecursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            foreach (var filter in filters)
            {
                filesFound.AddRange(Directory.GetFiles(searchFolder, String.Format("*.{0}", filter), searchOption));
            }
            return filesFound.ToArray();
        }
        static void Main(string[] args)
        {

            String searchFolder = @"C:\Users\Zhigonghui\source\repos\testDLL\testDLL\1000 dataset";
            //String searchFolder = @"C:\Users\Zhigonghui\source\repos\testDLL\testDLL\imgs";
            var filters = new String[] { "jpg", "jpeg", "png", "gif", "tiff", "bmp", "svg" };
            var files = GetFilesFrom(searchFolder, filters, false);

            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();
            
            TimeSpan ts = stopWatch.Elapsed;

            // USING EMGU.CV DEMO
            for (int num = 0; num < files.Length; num++)
            {
                Image<Gray, byte> img_input = new Image<Gray, byte>(files[num].ToString());
                detectQKT mydeteect = new detectQKT();
                ArrayList results_detection = new ArrayList();
                results_detection = mydeteect.DoDetection(img_input); // resulting image (index 0) , reuslting points location (index 1)

                Image<Gray, byte> img_measured = (Image<Gray, byte>)results_detection[0]; // cast Typle : Image<Gray, byte>
                ArrayList points = (ArrayList)results_detection[1]; // cast Typle : ArrayList

                img_measured.Save("C:/Users/Zhigonghui/source/repos/testDLL/testDLL/measured/"+num.ToString()+".JPG");
                //ImageViewer.Show(img_measured);
            }

            stopWatch.Stop();
            Debug.WriteLine("RunTime " + stopWatch.Elapsed);

            // USING MATLAB DLL DEMO
            /*   
            MLApp.MLApp matlab = new MLApp.MLApp();
            matlab.Execute(@"cd D:\工业视觉项目\BS翘扣头\翘扣头图像");
            object result = null;
            matlab.Feval("imread", 1, out result, "D:/工业视觉项目/BS翘扣头/翘扣头图像/NGs2/120608251400_1_2.jpg");
            var res = result as object[];
            byte[,] img_byte = (byte[,])res[0];

            /*
            int width = img.GetLength(0);
            int height = img.GetLength(1);
            int stride = width * 4;

            double[,] integers = new double[width, height];

            for (int x = 0; x < width; ++x)
            {
                for (int y = 0; y < height; ++y)
                {
                    integers[x, y] = img[x,y];
                }
            }
            
            //            object results_output = null;
            //            matlab.Feval("MWdetectBoudary", 2, out results_output, img);



            QKTclass mydetect = new QKTclass();
            MWArray[] resultsarrray = mydetect.MWDLLDetectQKT(2, (MWNumericArray)img_byte);
            MWNumericArray _output0 = (MWNumericArray)resultsarrray[0];
            MWNumericArray _output1 = (MWNumericArray)resultsarrray[1];
            Array byte_outputIMG = _output0.ToArray();
            Array byte_outputPoints = _output1.ToArray();

            */


        }
    }
}
