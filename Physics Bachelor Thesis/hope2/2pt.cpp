#include<iostream>
#include<fstream>
#include<array>
#include<cmath>
#include<iomanip>

//CODE FOR CALCULATING THE 2PT ANGLE


using namespace std;

int main()
{

//constants
const double pi=3.14159265358979323846264338327950288419716939937;

//array for input
array<double,4> a;
array<double,4> b;

//variables
double pt2;
int countline=2;//counter to compare pairs so as to be able to know what pairs to take into account, so no dublicates arise
double ra,dec;
double RA,DEC;
double x1,x2,y1,y2,z1,z2;
array<double,28> pt;
for (int i=0;i<28;i++)
	{
	pt[i]=0;
	}
long double countN=0;// counter-will become the total number of pairs


//streams
ifstream in1{"/home/tzikos/Desktop/hope2/2pt-lists/30-70-1-new.txt"};
ofstream out{"/home/tzikos/Desktop/hope2/results/2pt-30-70-1-bd-new.txt"}; 

//process for calculating 2pt
for (int i=1;i<=230;i++)
	{
	for(int j=0;j<4;j++)
		{
		in1>>a[j];
                }
	ra=a[0]*pi/180;
	dec=a[1]*pi/180;
	//vector1
	x1=sin(pi/2-dec)*cos(ra);
	y1=sin(pi/2-dec)*sin(ra);
	z1=cos(pi/2-dec);
	ifstream in2{"/home/tzikos/Desktop/hope2/2pt-lists-aux/30-70-1-new-aux.txt"};//you need to create a duplicate of the file to be able to open two instances of the same data at the same time.		
	for (int w=1;w<=231;w++)
		{			
		for(int l=0;l<4;l++)
			{
			in2>>b[l];
			}
		if (w>=countline)
			{
			countN++;
			RA=b[0]*pi/180;
			DEC=b[1]*pi/180;
			//vector2
			x2=sin(pi/2-DEC)*cos(RA);
			y2=sin(pi/2-DEC)*sin(RA);
			z2=cos(pi/2-DEC);
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
	out<<q+3<<" "<<pt[q]<<"\n";
	}

cout<<"countN = "<<countN;
}
			
