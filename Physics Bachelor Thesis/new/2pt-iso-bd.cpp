#include<iostream>
#include<fstream>
#include<array>
#include<cmath>

//CODE FOR CALCULATING THE 2PT ANGLE CUMULATION IN RESPECT TO EGREES IN THE ISOTROPIC DATA BEFORE DEPROPAGATION, AS WAS IN THE AUGER TXT DATA FILE

using namespace std;

int main()
{

//constants
const double pi=3.14159265358979323846264338327950288419716939937;

//array for input
array<double,5> a;
array<double,5> b;
char w1,w2;

//variables
double pt2;
int countline=2;
double longg,lat;
double LONG,LAT;
double x1,x2,y1,y2,z1,z2;
array<double,28> pt{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
long double countN=0;





//streams
ifstream in1{"/home/tzikos/Desktop/new/augerIn.txt"};
ofstream out{"/home/tzikos/Desktop/new/2pt-iso-bd.txt"}; 


for (int i=1;i<=81700;i++)
	{
	in1>>w1;
	for(int j=0;j<5;j++)
		{
		in1>>a[j];
                }
	longg=a[1]*pi/180;
	lat=a[2]*pi/180;
	//vector1
	x1=sin(pi/2-lat)*cos(longg);
	y1=sin(pi/2-lat)*sin(longg);
	z1=cos(pi/2-lat);
	ifstream in2{"/home/tzikos/Desktop/new/augerIn-aux.txt"};			
	for (int w=1;w<=81701;w++)
		{		
		in2>>w2;			
		for(int l=0;l<5;l++)
			{
			in2>>b[l];
			}
		if (w>=countline)
			{
			countN++;
			LONG=b[1]*pi/180;
			LAT=b[2]*pi/180;
			//vector2
			x2=sin(pi/2-LAT)*cos(LONG);
			y2=sin(pi/2-LAT)*sin(LONG);
			z2=cos(pi/2-LAT);
			//dot product of vectors-2pt correlation value
			pt2=acos(x1*x2+y1*y2+z1*z2);
			pt2=pt2*180/pi;
			for (int q=0;q<=27;q++)
				{
				if (pt2<=(q+3))
					{
					pt[q]++;
					}	
				}
			}		
		}	
	countline++;
	}

for (int q=0;q<=27;q++)
	{
	pt[q]=pt[q]*100/countN;
	}

for (int q=0;q<=27;q++)
	{
	out<<q+3<<" "<<pt[q]<<"\n";
	}
cout<<"\n"<<countN<<"\n";
}
