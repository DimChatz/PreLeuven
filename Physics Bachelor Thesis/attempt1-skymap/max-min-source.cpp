#include<iostream>
#include<fstream>
#include<array>


using namespace std;

int main()
{
ifstream in{"/home/tzikos/Desktop/attempt1-skymap/diff-calc-2pt.txt"};
array<double,6> a;
double max1=-180,max2=-90;
double min1=90,min2=90;
for (int i=1;i<=162;i++)
	{
	for (int j=0;j<6;j++)
		{
		in>>a[j];
		}
	if (a[3]>=max1)
		{max1=a[3];}
	if (a[3]<min1)
		{min1=a[3];}
	if (a[4]>max2)
		{max2=a[4];}
	if (a[4]<min2)
		{min2=a[4];}
	}
cout<<"\n"<<(max1-min1)/2<<" "<<(max2-min2)/2<<"\n";
cout<<"the center is at "<<(max1+min1)/2<<" RA and "<<(max2+min2)/2<<" DEC"<<"\n";
}
