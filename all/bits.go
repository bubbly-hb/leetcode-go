package all

// 位运算

// bits.Len(uint) 计算二进制位数
// bits.TrailingZeros(uint) 返回尾随零位的位数 注意 bits.TrailingZeros(0) 为uint的size，本机为64
// bits.OnesCount(uint) 返回二进制中1的个数

/*
strconv包
Itoa()：整型转字符串
func Itoa(i int) string

Atoi()：字符串转整型
func Atoi(s string) (i int, err error)

ParseBool() 函数用于将字符串转换为 bool 类型的值，它只能接受 1、0、t、f、T、F、true、false、True、False、TRUE、FALSE，
其它的值均返回错误，函数签名如下
func ParseBool(str string) (value bool, err error)

ParseInt() 函数用于返回字符串表示的整数值（可以包含正负号），函数签名如下
func ParseInt(s string, base int, bitSize int) (i int64, err error)
参数说明：
base 指定进制，取值范围是 2 到 36。如果 base 为 0，则会从字符串前置判断，“0x”是 16 进制，“0”是 8 进制，否则是 10 进制。
bitSize 指定结果必须能无溢出赋值的整数类型，0、8、16、32、64 分别代表 int、int8、int16、int32、int64。
返回的 err 是 *NumErr 类型的，如果语法有误，err.Error = ErrSyntax，如果结果超出类型范围 err.Error = ErrRange。

ParseUint() 函数的功能类似于 ParseInt() 函数，但 ParseUint() 函数不接受正负号，用于无符号整型，函数签名如下：
func ParseUint(s string, base int, bitSize int) (n uint64, err error)

ParseFloat() 函数用于将一个表示浮点数的字符串转换为 float 类型，函数签名如下。
func ParseFloat(s string, bitSize int) (f float64, err error)
参数说明：
如果 s 合乎语法规则，函数会返回最为接近 s 表示值的一个浮点数（使用 IEEE754 规范舍入）。
bitSize 指定了返回值的类型，32 表示 float32，64 表示 float64；
返回值 err 是 *NumErr 类型的，如果语法有误 err.Error=ErrSyntax，如果返回值超出表示范围，返回值 f 为 ±Inf，err.Error= ErrRange。

FormatBool() 函数可以一个 bool 类型的值转换为对应的字符串类型，函数签名如下。
func FormatBool(b bool) string

FormatInt() 函数用于将整型数据转换成指定进制并以字符串的形式返回，函数签名如下
func FormatInt(i int64, base int) string
其中，参数 i 必须是 int64 类型，参数 base 必须在 2 到 36 之间，返回结果中会使用小写字母“a”到“z”表示大于 10 的数字

FormatUint() 函数与 FormatInt() 函数的功能类似，但是参数 i 必须是无符号的 uint64 类型，函数签名如下
func FormatUint(i uint64, base int) string

FormatFloat() 函数用于将浮点数转换为字符串类型，函数签名如下：
func FormatFloat(f float64, fmt byte, prec, bitSize int) string
参数说明：
bitSize 表示参数 f 的来源类型（32 表示 float32、64 表示 float64），会据此进行舍入。
fmt 表示格式，可以设置为“f”表示 -ddd.dddd、“b”表示 -ddddp±ddd，指数为二进制、“e”表示 -d.dddde±dd 十进制指数、
	“E”表示 -d.ddddE±dd 十进制指数、“g”表示指数很大时用“e”格式，否则“f”格式、“G”表示指数很大时用“E”格式，否则“f”格式。
prec 控制精度（排除指数部分）：当参数 fmt 为“f”、“e”、“E”时，它表示小数点后的数字个数；当参数 fmt 为“g”、“G”时，它控制总的数字个数。
	如果 prec 为 -1，则代表使用最少数量的、但又必需的数字来表示 f。
*/

// 按位考虑，分别考虑答案的每一个二进制位是啥
// 只出现一次的数字 https://leetcode.cn/problems/WGki4K/?envType=study-plan&id=lcof-ii&plan=lcof&plan_progress=ckvikye
// func singleNumber(nums []int) int {
//     ans := int32(0)
//     for i := 0; i < 32; i++ {
//         s := int32(0)
//         for _, a := range nums {
//             s += int32(a>>i&1)
//         }
//         if s % 3 > 0 {
//             ans |= 1 << i
//         }
//     }
//     return int(ans)
// }

// 判断两个字符串是否包含相同字符，可以使用位掩码的最低 26 位分别表示每个字母是否在这个单词中出现 然后两个字符串对应的位掩码&结果为0则不包含
// 单词长度的最大乘积  https://leetcode.cn/problems/aseY1I/
