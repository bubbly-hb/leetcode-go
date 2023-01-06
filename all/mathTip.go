package all

// 对于每一个三角形，我们尝试寻找第四个点并判定它们是否能形成一个矩形。
// 假设前三个点分别是 p1, p2, p3，并且 p2 与 p3 在最终的矩形中处于对角位置。那么第四个点一定是 p4 = p2 + p3 - p1（向量计算），
// 因为 p1, p2, p4, p3 一定形成一个平行四边形，满足 p1 + (p2 - p1) + (p3 - p1) = p4。
const MOD int = 1e9 + 7

func quick_pow(x, y int) int {
	base, ans := x, 1
	for y > 0 {
		if y&1 == 1 {
			ans = ans * base
		}
		base *= base
		y >>= 1
	}
	return ans
}
func quick_pow_mod(x, y, MOD int) int {
	base, ans := x%MOD, 1
	for y > 0 {
		if y&1 == 1 {
			ans = ans * base % MOD
		}
		base = base * base % MOD
		y >>= 1
	}
	return ans
}

// 快速幂取模例题：数组元素的最小非零乘积   https://leetcode.cn/problems/minimum-non-zero-product-of-the-array-elements/

// 裴蜀定理（或贝祖定理）得名于法国数学家艾蒂安·裴蜀，说明了对任何整数a、b和它们的最大公约数d，关于未知数x和y的线性不定方程（称为裴蜀等式）：
// 若a,b是整数,且gcd(a,b)=d，那么对于任意的整数x,y,ax+by都一定是d的倍数，特别地，一定存在整数x,y，使ax+by=d成立。
// 它的一个重要推论是：a,b互质的充分必要条件是存在整数x,y使ax+by=1.
// 例题：检查好数组  https://leetcode.cn/problems/check-if-it-is-a-good-array/

// 分解质因数
func distinctPrimeFactors(v int) (ans []int) {
	for i := 2; i*i <= v; i++ {
		if v%i == 0 {
			ans = append(ans, i)
			for ; v%i == 0; v /= i {
			}
		}
	}
	if v > 1 {
		ans = append(ans, v)
	}
	return
}

// 例题：数组乘积中的不同质因数数目 https://leetcode.cn/problems/distinct-prime-factors-of-product-of-array/

// 素数  不存在长度为8的素数
func primeCheck(x int) bool {
	if x < 2 {
		return false
	}
	if x == 2 || x == 3 {
		return true
	}
	for i := 2; i*i <= x; i++ {
		if x%i == 0 {
			return false
		}
	}
	return true
}

// 埃氏筛素数
const mx = 1e6 + 1

var primes = make([]int, 0, mx)

func init() {
	np := [mx]bool{}
	for i := 2; i < mx; i++ {
		if !np[i] {
			for j := i * i; j < mx; j += i {
				np[j] = true
			}
			primes = append(primes, i)
		}
	}
}

// 线性筛（欧拉筛）
// const mx = 1e6 + 1

// var primes = make([]int, 0, mx)

// func init() {
// 	np := [mx]bool{}
// 	for i := 2; i < mx; i++ {
// 		if !np[i] {
// 			primes = append(primes, i)
// 		}
// 		for _, p := range primes {
// 			if i*p >= mx {
// 				break
// 			}
// 			np[i*p] = true
// 			if i%p == 0 {
// 				break
// 			}
// 		}
// 	}
// }

// 有重复元素的全排列
// 给定k个元素，其中第i个元素有ai个，总个数为n，则全排列个数为： n! / (a1! * a2! * a3! * ... * ak!)

// 费马小定理
// 若 p 为素数，gcd(a, p) == 1，则 a^(p - 1) ≡ 1 (mod p)。
// 另一个形式：对于任意整数 a，有 a^p ≡ a (mod p)。

// 组合计数与费马小定理例题：
// 统计同位异构字符串数目  https://leetcode.cn/problems/count-anagrams/
