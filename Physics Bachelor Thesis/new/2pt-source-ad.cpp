#include<iostream>
#include<fstream>
#include<array>
#include<cmath>

//CODE FOR CALCULATING THE 2PT ANGLE AFTER THE DEPROPAGATION OF THE SOURCE

using namespace std;

int main()
{

//constants
const double pi=3.14159265358979323846264338327950288419716939937;

//array for input
array<double,3> a;
array<double,3> b;

//variables
double pt2;
int countline=2;
double longg,lat;
double LONG,LAT;
double x1,x2,y1,y2,z1,z2;
array<double,28> pt{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
long double countN=0;





//streams
ifstream in1{"/home/tzikos/Desktop/new/CentauriA-after.txt"};
ofstream out{"/home/tzikos/Desktop/new/2pt-source-ad.txt"}; 


for (int i=1;i<=296;i++)
	{
	for(int j=0;j<3;j++)
		{
		in1>>a[j];
                }
	longg=a[1]*pi/180;
	lat=a[2]*pi/180;
	//vector1
	x1=sin(pi/2-lat)*cos(longg);
	y1=sin(pi/2-lat)*sin(longg);
	z1=cos(pi/2-lat);
	ifstream in2{"/home/tzikos/Desktop/new/CentauriA-aux-after.txt"};			
	for (int w=1;w<=297;w++)
		{					
		for(int l=0;l<3;l++)
			{
			in2>>b[l];
			}
		if (w>=countline)
			{
			countN++;
			LONG=b[1]*pi/180;
			LAT=b[2]*pi/180;
			//cout<<b[2]<<"\n"; //checkin
			//vector2
			x2=sin(pi/2-LAT)*cos(LONG);
			y2=sin(pi/2-LAT)*sin(LONG);
			z2=cos(pi/2-LAT);
			//dot product of vectors-2pt correlation value
			pt2=acos(x1*x2+y1*y2+z1*z2);
			pt2=pt2*180/pi;
			if (w==2)
				{
				cout<<pt2<<"\n";
				cout<<(x1*x2)<<" "<<(y1*y2)<<" "<<(z1*z2)<<"\n"; //check if all of those are positives as they should be
				cout<<x1<<" "<<y1<<" "<<z1<<"\t"<<x2<<" "<<y2<<" "<<z2<<"\n";	//check to see if he calculates coordinates correctly #not
				cout<<a[1]<<" "<<a[2]<<"\t"<<b[1]<<" "<<b[2]<<"\n";	//check if its getting input correctly
				cout<<sin(pi/2+19.94*pi/180)*cos(-46.01*pi/180)<<"\n";
				}
			for (int q=0;q<=27;q++)
				{
				if (pt2<=(q+3))
					{
					pt[q]++;
					//cout<<pt2<<"\n";//checkin to see that no number upwards of 15 arises, the lisst contains angular differences up to 15 degrees.
					}	
				}
			}		
		}
	in2.close();	
	countline++;
	}
for (int q=0;q<=27;q++)
	{
	out<<q+3<<" "<<pt[q]<<"\n";
	}
cout<<"countN = "<<countN;
}
