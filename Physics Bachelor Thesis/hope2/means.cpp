#include<iostream>
#include<array>
#include<fstream>
#include<cmath>

//CODE FOR FINDING THE MEAN AND SIGMA OF THE 2PT ANGLE IN ORDER TO MAKE THEN PLOTS

using namespace std;

int main()
{
//streams for inputing data
ifstream in1{"/home/tzikos/Desktop/hope2/results/2pt-30-70-1-bd-new.txt"};
ifstream in2{"/home/tzikos/Desktop/hope2/results/2pt-30-70-2-bd-new.txt"};
ifstream in3{"/home/tzikos/Desktop/hope2/results/2pt-30-70-3-bd-new.txt"};
ifstream in4{"/home/tzikos/Desktop/hope2/results/2pt-30-70-4-bd-new.txt"};
ifstream in5{"/home/tzikos/Desktop/hope2/results/2pt-30-70-5-bd-new.txt"};
ifstream in6{"/home/tzikos/Desktop/hope2/results/2pt-30-70-6-bd-new.txt"};
ifstream in7{"/home/tzikos/Desktop/hope2/results/2pt-30-70-7-bd-new.txt"};
ifstream in8{"/home/tzikos/Desktop/hope2/results/2pt-30-70-8-bd-new.txt"};
ifstream in9{"/home/tzikos/Desktop/hope2/results/2pt-30-70-9-bd-new.txt"};
ifstream in10{"/home/tzikos/Desktop/hope2/results/2pt-30-70-10-bd-new.txt"};

//stream for output
ofstream out{"/home/tzikos/Desktop/hope2/means/2pt-mean-30-70-bd-new.txt"};

//arrays for data
array<double,2> a;//array for reading data
array<array<double,28>,10> b;//array for storing data
array<double,28> mean;//will store means
array<double,28> sigma;//same for sigmas

//START OF PROCCESS
//for in1
for (int i=0;i<28;i++)
	{
	for (int j=0;j<2;j++)
		{
		in1>>a[j];
		}
	b[0][i]=a[1];
	}

//for in2
for (int i=0;i<28;i++)
	{
	for (int j=0;j<2;j++)
		{
		in2>>a[j];
		}
	b[1][i]=a[1];
	}

//for in3
for (int i=0;i<28;i++)
	{
	for (int j=0;j<2;j++)
		{
		in3>>a[j];
		}
	b[2][i]=a[1];
	}

//for in4
for (int i=0;i<28;i++)
	{
	for (int j=0;j<2;j++)
		{
		in4>>a[j];
		}
	b[3][i]=a[1];
	}

//for in5
for (int i=0;i<28;i++)
	{
	for (int j=0;j<2;j++)
		{
		in5>>a[j];
		}
	b[4][i]=a[1];
	}

//for in6
for (int i=0;i<28;i++)
	{
	for (int j=0;j<2;j++)
		{
		in6>>a[j];
		}
	b[5][i]=a[1];
	}

//for in7
for (int i=0;i<28;i++)
	{
	for (int j=0;j<2;j++)
		{
		in7>>a[j];
		}
	b[6][i]=a[1];
	}

//for in8
for (int i=0;i<28;i++)
	{
	for (int j=0;j<2;j++)
		{
		in8>>a[j];
		}
	b[7][i]=a[1];
	}

//for in9
for (int i=0;i<28;i++)
	{
	for (int j=0;j<2;j++)
		{
		in9>>a[j];
		}
	b[8][i]=a[1];
	}

//for in10
for (int i=0;i<28;i++)
	{
	for (int j=0;j<2;j++)
		{
		in10>>a[j];
		}
	b[9][i]=a[1];
	}

//calculation of mean
for (int i=0;i<28;i++)
	{	
	for (int j=0;j<10;j++)
		{
		mean[i]+=b[j][i];
		}
	mean[i]/=10;
	}


//calculation of sigma
for (int i=0;i<28;i++)
	{	
	for (int j=0;j<10;j++)
		{
		sigma[i]+=pow((b[j][i]-mean[i]),2);
		}
	sigma[i]=sqrt(sigma[i]/9);
	}

//output to file
for (int i=0;i<28;i++)
	{
	out<<i+3<<" "<<mean[i]<<" "<<sigma[i]<<"\n";
	}
}
