#include<iostream>
#include<array>
#include<fstream>
#include<cmath>

using namespace std;

int main()
{
ifstream in{"/home/tzikos/Desktop/attempt1/final-list-high-E.txt"};
ofstream out{"/home/tzikos/Desktop/attempt1/diff-calc.txt"};

double dRA,dDEC;

array<double,6> a;

for (int i=0;i<81701;i++)
	{
	for (int j=0;j<6;j++)
		{	
		in>>a[j];
		}
	dRA=a[4]-a[2];
	dDEC=a[5]-a[3];
	if (((fabs(dRA)>30)and(fabs(dRA)<180)) or ((fabs(dDEC)>30)and(fabs(dDEC)<70)))
		{
		out<<a[0]<<" "<<a[2]<<" "<<a[3]<<" "<<dRA<<" "<<dDEC<<"\n";
		}
	}

}
