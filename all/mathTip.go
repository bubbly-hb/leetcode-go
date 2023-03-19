package all

import "math/big"

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

// 欧拉回路
// 如果图G中的一个路径包括每个边恰好一次，则该路径称为欧拉路径(Euler path)。
// 如果一个回路是欧拉路径，则称为欧拉回路(Euler circuit)。
// 具有欧拉回路的图称为欧拉图（简称E图）。具有欧拉路径但不具有欧拉回路的图称为半欧拉图。
// 无向图存在欧拉回路的充要条件：
// 一个无向图存在欧拉回路，当且仅当该图所有顶点度数都为偶数,且该图是连通图。
// 有向图存在欧拉回路的充要条件：
// 一个有向图存在欧拉回路，所有顶点的入度等于出度且该图是连通图。

// 求解欧拉回路例题：
// 破解保险箱 https://leetcode.cn/problems/cracking-the-safe/     dfs 或者 贪心构造
/*
func crackSafe(n int, k int) string {
	dic := map[int]bool{}
	bound := int(math.Pow(float64(k), float64(n-1)))
	ans := ""

	var dfs func(int)
	dfs = func(node int) {
		for i := 0; i < k; i++ {
			tn := node*k + i // 不能写成 node = node * k + i  因为node在后面的循环会用到原来的值
			if !dic[tn] {
				dic[tn] = true
				dfs(tn % bound)
				ans += strconv.Itoa(i) // 相当于一直深搜把所有边都走完了然后反向添加答案
			}
		}
	}

	dfs(0)
	for i := 1; i < n; i++ {
		ans += "0"
	}
	return ans
}

func crackSafe(n int, k int) string { // k进制去除最高位，直接mod
	ans := []byte{}
	bound := int(math.Pow(float64(k), float64(n-1)))
	edges := make([]int, bound)
	for i := range edges {
		edges[i] = k - 1
	}
	node := 0
	for edges[node] >= 0 {
		edge := edges[node]
		edges[node]--
		node = (node*k + edge) % bound
		ans = append(ans, byte('0'+edge))
	}
	pre := make([]byte, n-1)
	for i := 0; i < n-1; i++ {
		pre[i] = '0'
	}
	return string(pre) + string(ans)
}
*/

/*
// 最大公因数gcd
func gcd(x, y int) int {
	if x%y == 0 {
		return y
	}
	return gcd(y, x%y)
}

// 最小公倍数lcm
func lcm(x, y int) int {
	return x * y / gcd(x, y)
}

gcd, lcm例题： 丑数III    https://leetcode.cn/problems/ugly-number-iii/
*/

// 序列中不同最大公约数的数目 枚举最大公约数 + 循环优化
// https://leetcode.cn/problems/number-of-different-subsequences-gcds/
func countDifferentSubsequenceGCDs(nums []int) (ans int) {
	mx := 0
	for _, v := range nums {
		if v > mx {
			mx = v
		}
	}
	has := make([]bool, mx+1)
	for _, v := range nums {
		if !has[v] {
			ans++
		}
		has[v] = true
	}
	for i := 1; i <= mx/3; i++ {
		if has[i] {
			continue
		}
		g := 0                                      // 0 和任何数 x 的最大公约数都是 x
		for j := i * 2; j <= mx && g != i; j += i { // 枚举 i 的倍数 j
			if has[j] { // 如果 j 在 nums 中
				g = gcd(g, j) // 更新最大公约数
			}
		}
		if g == i { // 找到一个答案
			ans++
		}
	}
	return
}
func gcd(x, y int) int {
	if x%y == 0 {
		return y
	}
	return gcd(y, x%y)
}

// 二维差分
// 子矩阵元素加 1   https://leetcode.cn/problems/increment-submatrices-by-one/
func rangeAddQueries(n int, queries [][]int) [][]int {
	// 差分数组
	diff := make([][]int, n+1)
	for i := range diff {
		diff[i] = make([]int, n+1)
	}
	for _, q := range queries {
		a, b, c, d := q[0], q[1], q[2]+1, q[3]+1
		diff[a][b]++
		diff[a][d]--
		diff[c][b]--
		diff[c][d]++
	}

	// 对diff求二维前缀和即为变化量
	ans := make([][]int, n+1)
	for i := range ans {
		ans[i] = make([]int, n+1)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			ans[i+1][j+1] = ans[i][j+1] + ans[i+1][j] - ans[i][j] + diff[i][j]
		}
	}

	// 因为原数组全为0，所以变化量即为最终值
	ans = ans[1:]
	for i := range ans {
		ans[i] = ans[i][1:]
	}
	return ans
}

// 格雷编码
// https://leetcode.cn/problems/gray-code/
func grayCode(n int) []int {
	ans := make([]int, 1, 1<<n)
	for i := 1; i <= n; i++ {
		for j := len(ans) - 1; j >= 0; j-- {
			ans = append(ans, ans[j]^(1<<(i-1)))
		}
	}
	return ans
}

func grayCode2(n int) []int { // 格雷编码公式法
	ans := make([]int, 1<<n)
	for i := range ans {
		ans[i] = i>>1 ^ i
	}
	return ans
}

// 格雷编码其他例题：
// 循环码排列 https://leetcode.cn/problems/circular-permutation-in-binary-representation/ 指定以某个数开头的话在算的过程中异或这个数即可

// 路径的数目 二项式定理  https://leetcode.cn/problems/unique-paths/
func uniquePaths(m int, n int) int {
	return int(new(big.Int).Binomial(int64(m+n-2), int64(m-1)).Int64())
}
