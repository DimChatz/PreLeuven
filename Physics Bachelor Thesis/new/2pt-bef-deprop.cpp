#include<iostream>
#include<fstream>
#include<array>
#include<cmath>
#include<iomanip>

//CODE FOR CALCULATING THE 2PT ANGLE FROM THE !!INPUT!! OF CRT, BEFORE DEPROPAGATION


using namespace std;

int main()
{

//constants
const double pi=3.14159265358979323846264338327950288419716939937;

//array for input
array<double,5> a;
array<double,5> b;

//variables
double pt2;
int countline=2;//counter to compare pairs so as to be able to know what pairs to take into account, so no dublicates arise
double longg,lat;
double LONG,LAT;
double x1,x2,y1,y2,z1,z2;
array<double,28> pt{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
long double countN=0;// counter-will become the total number of pairs
char A,B;


//streams
ifstream in1{"/home/tzikos/Desktop/new/temp/In-50-50-1.txt"};
ofstream out{"/home/tzikos/Desktop/new/results/50-50/2pt-bd-50-50-1.txt"}; 

//process for calculating 2pt
for (int i=1;i<=230;i++)
	{
	in1>>A;
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
	ifstream in2{"/home/tzikos/Desktop/new/temp/In-aux-50-50-1.txt"};//you need to create a duplicate of the file to be able to open two instances of the same data at the same time.		
	for (int w=1;w<=231;w++)
		{		
		in2>>B;			
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
					//cout<<pt2<<"\n";  //for checking
					}	
				}
			}		
		}	
	countline++;
	}
for (int q=0;q<=27;q++)
	{
	out<<q+3<<" "<<pt[q]<<"\n";
	}

cout<<"countN = "<<countN;
}
			
