// DiffNum.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <DiffMath.h>
#include <Vec.h>

using namespace DiffNum;

int main()
{
    DiffDouble a(2., 2, 0);
    DiffDouble b(3., 2, 1);
    Vec<DiffDouble, 3> v1, v2;
    v1[0] = 8.7;
    v1[1] = 4.3;
    v1[2] = 7.;
    v1[2].SetVar(1, 0);

    v2[0] = -6.7;
    v2[1] = 4.1;
    v2[2] = 2.3;
    auto v3 = Vec<DiffDouble, 3>::dot(v1, v2);

    auto c = a + b;
    auto d = Log(Sin(a / c));
    std::cout << v3 << std::endl;
    return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
