using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.Structure;
using System.Diagnostics;
using System.Collections;
using Emgu.CV.UI;

namespace testDLL
{
    class detectQKT
    {
        public ArrayList DoDetection(Image<Gray, byte> img_input)
        {
            ArrayList result_output = new ArrayList();
            Image<Gray, byte> img1 = img_input;

            Image<Gray, byte> img_normal = img1;    // for normal image bitmap
            Image<Gray, byte> img_obvious = img1;   // for obvious image bitmap

            int counter_normalQKT = 0;
            int counter_obviousQKT = 0;
            int counter_CLP = 0;
            /* Canny Edge */
            Image<Gray, byte> Img_Otsu_Gray = img1.CopyBlank();
            Image<Gray, byte> Img_edge_BW = img1.CopyBlank();
            double CannyAccThresh = CvInvoke.Threshold(img1, Img_Otsu_Gray, 0, 255, Emgu.CV.CvEnum.ThresholdType.Otsu);
            double CannyThresh = 0.4*CannyAccThresh;

            img1 = img1.SmoothGaussian(5);
            Img_edge_BW = img1.Canny(CannyThresh, CannyAccThresh*1);   // setting for Canny1
            //ImageViewer.Show(Img_edge_BW);

            /* Morphology */
            Mat dilateKernel = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(5, 5), new Point(-1, -1));
            Image<Gray, byte> Img_dilate = Img_edge_BW.CopyBlank();
            Img_dilate = Img_edge_BW.MorphologyEx(Emgu.CV.CvEnum.MorphOp.Dilate, dilateKernel,new Point(-1, -1),1, Emgu.CV.CvEnum.BorderType.Default, new MCvScalar(1.0));
            //ImageViewer.Show(Img_dilate);


            int row = Img_dilate.Rows;
            int col = Img_dilate.Cols;
            
            // find the dense bound points for coordinating the corner point
            ArrayList topBound = new ArrayList();
            for (int c= (col/3); c<col; c++)  // take half of the bound for analysis
            {
                for (int r = 0; r < row; r++)
                {
                    double thispixelvalue = Img_dilate.Data[r,c,0]/255;
                    //Debug.WriteLine(r.ToString()+','+c.ToString()+','+thispixelvalue.ToString());
                    if (thispixelvalue == 1)
                    {
                        int[] thispoint = { c, r };
                        topBound.Add(thispoint);
                        break;
                    }
                }
            }

            // calculate curvature and derivative for coordinating the corner point
            ArrayList curvatures = new ArrayList();
            ArrayList normals = new ArrayList();
            ArrayList firstboudID = new ArrayList();
            ArrayList secondboudID = new ArrayList();

            ArrayList thefirst = new ArrayList();
            ArrayList theSecond = new ArrayList();


            int[] boarder1 = (int[]) topBound[0];
            int[] boarder2 = (int[]) topBound[topBound.Count-1];
            topBound.Insert(0, boarder1);
            topBound.Insert(0, boarder1);
            topBound.Add(boarder2);
            topBound.Add(boarder2);

            for (int num = topBound.Count*4/5; num < topBound.Count-2; num++)
            {
                int[] thePoint = (int[]) topBound[num];

                int[] prvpoint = (int[])topBound[num - 1];
                int[] nextpoint = (int[])topBound[num + 1];
                int[] prepoint1 = (int[])topBound[num - 2];
                int[] nextpoint1 = (int[])topBound[num + 2];

                double thePoint_xd1 = nextpoint[0] - prvpoint[0];
                double thePoint_yd1 = nextpoint[1] - prvpoint[1];

                double thePoint_xd2 = nextpoint1[0] + prepoint1[0] - 2 * thePoint[0];
                double thePoint_yd2 = nextpoint1[1] - prepoint1[1] - 2 * thePoint[1];

                double k1 = Math.Abs(thePoint_xd1 * thePoint_yd2 - thePoint_yd1 * thePoint_xd2);
                double k2 = Math.Pow(Math.Sqrt(Math.Pow(thePoint_xd2,2) + Math.Pow(thePoint_yd2, 2)),3);
                double theK = k1 / k2;
                //Debug.WriteLine("k: " + theK.ToString());
                curvatures.Add(theK);
                // cal normal
                double dx = nextpoint[0] - thePoint[0];
                double dy = nextpoint[1] - thePoint[1];
                double[] thisnormal = {-dy, dx};
                normals.Add(thisnormal);


                if (theK > 0 && thisnormal[0] * thisnormal[1] < 0)
                {
                    firstboudID.Add(num);
                    thefirst.Add(thePoint);
                    int diff = (int)thePoint_yd1;
                    //Debug.WriteLine("2nd: " + diff.ToString());
                    if (diff > 1)
                    {
                        secondboudID.Add(num);
                        theSecond.Add(thePoint);
                    }
                }

            }
            int[] cornerpoint;
            if (secondboudID.Count == 0)
            {
                counter_CLP++;
                int index = (int) firstboudID[firstboudID.Count - 1];
                int[] points = (int[])topBound[index];
                cornerpoint = new int[] { points[0], points[1] };
            }
            else
            {
                int index = (int)secondboudID[0];
                int[] points = (int[])topBound[index];
                cornerpoint = new int[] { points[0], points[1] };
            }
            


            // finding the edge accroding to the corner point
            double CannyAccThresh1 = CvInvoke.Threshold(img1, Img_Otsu_Gray, 0, 255, Emgu.CV.CvEnum.ThresholdType.Otsu);
            double CannyThresh1 = 0.5 * CannyAccThresh1;

            Image<Gray, byte> Img_bound_BW = img1.Canny(CannyThresh1, CannyAccThresh1);      //setting for Canny2
            //ImageViewer.Show(Img_bound_BW);

            int row1 = Img_bound_BW.Rows;
            int col1 = Img_bound_BW.Cols;
            int X = cornerpoint[0];
            int Y = cornerpoint[1];
            int counter = 0;
            ArrayList bound = new ArrayList();
            bound.Add(new int[] { X,Y});
            for (int c = 20; c < X-1; c=c+20)
            {
                int column = X - c;
                bool isFound = false;
                for (int r = Y-50; r < Y+50; r++)
                {
                    double thispixelvalue = Img_bound_BW.Data[r, column, 0] / 255;

                    if (thispixelvalue == 1)
                    {
                        int[] thispoint = { column, r };
                        bound.Add(thispoint);
                        for (int iter =1; iter <= counter; iter++)
                        {
                            int offset = r - Y;
                            int[] newvar = { column+20*iter, r - ((offset * (iter)) / (counter + 1))};
                            bound.Insert(bound.Count - iter-1, newvar);
                            bound.RemoveAt(bound.Count - iter-1);
                        }
                        isFound = true;
                        counter = 0;

                        break;
                    }
                }
                if (!isFound)
                {
                    counter++;
                    int[] lastRow = (int[])bound[bound.Count - 1];
                    int[] newvar = { column, (lastRow[1]+Y)/2 - counter };
                    bound.Add(newvar);
                }
            }

            // display results and save as img file
            Bitmap bmpTif = new Bitmap(img_normal.Bitmap);

            Bitmap bmp = bmpTif.Clone(new Rectangle(0, 0, img_normal.Cols, img_normal.Rows),
                System.Drawing.Imaging.PixelFormat.Format32bppRgb);

            Graphics g = Graphics.FromImage(bmp);
            g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;
            for(int num = 0; num < bound.Count; num++)
            {
                int[] thepoint = (int[])bound[num];
                int OFFSETRECALI = 15;
                g.DrawString("+", new Font("Tahoma", OFFSETRECALI), Brushes.White, new Point(thepoint[0]- OFFSETRECALI, thepoint[1]- OFFSETRECALI));
            }
            /*
            for (int num = 0; num < thefirst.Count; num++)
            {
                int[] thepoint = (int[])thefirst[num];
                int OFFSETRECALI = 15;
                g.DrawString("+", new Font("Tahoma", OFFSETRECALI), Brushes.Red, new Point(thepoint[0] - OFFSETRECALI, thepoint[1] - OFFSETRECALI));
            }
            
            for (int num = 0; num < theSecond.Count; num++)
            {
                int[] thepoint = (int[])theSecond[num];
                int OFFSETRECALI = 15;
                g.DrawString("+", new Font("Tahoma", OFFSETRECALI), Brushes.Green, new Point(thepoint[0] - OFFSETRECALI, thepoint[1] - OFFSETRECALI));
            }
            */


            //bmp.Save("C:/Users/Zhigonghui/source/repos/testDLL/testDLL/imgs/120526551700_2_1_measured.JPG");

            // extral process for obvious and normal

            ArrayList fullbound = new ArrayList();
            for (int c = 0; c < col; c=c+10)  // take half of the bound for analysis
            {
                for (int r = 0; r < row; r++)
                {
                    double thispixelvalue = Img_dilate.Data[r, c, 0] / 255;
                    //Debug.WriteLine(r.ToString()+','+c.ToString()+','+thispixelvalue.ToString());
                    if (thispixelvalue == 1)
                    {
                        int[] thispoint = { c, r };
                        fullbound.Add(thispoint);
                        break;
                    }
                }
            }

            int Tqiao = 35;
            int numberOfconcern = 2;

            int[] a = (int[])fullbound[0];
            int[] b = (int[])fullbound[1];
            double cornerpoints = (a[1] + b[1]) / numberOfconcern;
            int[] y = (int[])fullbound[fullbound.Count - 2];
            int[] z = (int[])fullbound[fullbound.Count - 1];
            double endpoints = cornerpoint[1];
            double isQiao = cornerpoints - endpoints;
            //Debug.WriteLine("ratio of QKT: "+ isQiao.ToString());


            ArrayList result_bound = new ArrayList();

            if (isQiao > Tqiao)
            {
                
                int index = (int)firstboudID[0];
                int[] initialpoint = (int[])topBound[index];

                for (int iter=0;iter< firstboudID.Count; iter++)
                {
                    int thisindex = (int)firstboudID[iter];
                    int[] thispoint = (int[])topBound[thisindex];

                    if (thispoint[1] < initialpoint[1])
                    {
                        initialpoint = thispoint;
                    }
                }

                int[] cornerpoint_Recal = initialpoint;
                int X_recal = cornerpoint_Recal[0];
                int Y_recal = cornerpoint_Recal[1];

                int counter_recal = 0;
                ArrayList bound_recal = new ArrayList();
                bound_recal.Add(new int[] { X_recal, Y_recal });
                for (int c = 20; c < X_recal - 1; c = c + 20)
                {
                    int column = X_recal - c;
                    bool isFound = false;
                    for (int r = 100; r < row; r++)
                    {
                        double thispixelvalue = Img_bound_BW.Data[r, column, 0] / 255;

                        if (thispixelvalue == 1)
                        {
                            int[] thispoint = { column, r };
                            bound_recal.Add(thispoint);
                            for (int iter = 0; iter < counter_recal; iter++)
                            {
                                int offset = r - Y_recal;
                                int[] newvar = { column + 20 * iter, r - ((offset * (iter)) / (counter + 1)) };
                                bound_recal.Insert(bound_recal.Count - iter - 1, newvar);
                                bound_recal.RemoveAt(bound_recal.Count - iter - 1);
                            }
                            isFound = true;
                            counter_recal = 0;

                            break;
                        }
                    }
                    if (!isFound)
                    {
                        counter++;
                        int[] lastRow = (int[])bound_recal[bound_recal.Count - 1];
                        int[] newvar = { column, lastRow[1] };
                        bound_recal.Add(newvar);
                    }
                }

                // display results and save as img file
                Bitmap bmpTif_recal = new Bitmap(img_obvious.Bitmap);

                Bitmap bmp_real = bmpTif.Clone(new Rectangle(0, 0, img_obvious.Cols, img_obvious.Rows),
                    System.Drawing.Imaging.PixelFormat.Format32bppRgb);

                Graphics g_recal = Graphics.FromImage(bmp_real);
                g_recal.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;
                for (int num = 0; num < bound_recal.Count; num++)
                {
                    int[] thepoint = (int[])bound_recal[num];
                    int OFFSETRECALI = 15;
                    g_recal.DrawString("+", new Font("Tahoma", OFFSETRECALI), Brushes.White, new Point(thepoint[0] - OFFSETRECALI, thepoint[1] - OFFSETRECALI));
                }

                Image<Gray, byte> result_obv = new Image<Gray, byte>(bmp_real);
                //ImageViewer.Show(result_obv);
                result_bound = bound_recal;
                //bmp_real.Save("C:/Users/Zhigonghui/source/repos/testDLL/testDLL/imgs/120526551700_2_1_measured.JPG");

                result_output.Add(result_obv);
                result_output.Add(bound_recal);

                counter_obviousQKT++;

            }
            else
            {
                Image<Gray, byte> result_normal = new Image<Gray, byte>(bmp);
                //ImageViewer.Show(result_normal);
                result_bound = bound;
                result_output.Add(result_normal);
                result_output.Add(bound);
                counter_normalQKT++;
            }

            /*
            Debug.WriteLine("Num of CLP: "+ counter_CLP.ToString());
            Debug.WriteLine("Num of Obvious: " + counter_obviousQKT.ToString());
            Debug.WriteLine("Num of Normal: " + counter_normalQKT.ToString());
            */

            return result_output;
        }
    }
}
