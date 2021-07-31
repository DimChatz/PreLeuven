#include<iostream>
#include<fstream>

// we will find the cr's that start from Centauri A and arrive at the Earth

using namespace std;


int main()
{

ifstream in1{"/home/tzikos/Desktop/new/CentauriA-after.txt"};
ofstream out{"/home/tzikos/Desktop/new/start-finish.txt"};

int a,b,count=0;

for (int i=0;i<300;i++)
	{
	in1>>a;
	ifstream in2{"/home/tzikos/Desktop/new/CentauriA-bef.txt"};
	for (int j=0;j<346;j++)
		{
		in2>>b;
		if (a==b)
			{
			out<<a<<"\n";
			}
		count++;
		}
	}
cout<<"\ncount = "<<count<<"\t"<<a<<"\t"<<b<<"\n";
}
