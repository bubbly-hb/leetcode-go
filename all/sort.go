package all

import (
	"math/rand"
)

// 快速排序
func QuickSort(a []int) {
	quickSort(a, 0, len(a)-1)
}

func quickSort(a []int, l, r int) {
	if l > r {
		return
	}
	p := findPos(a, l, r)
	quickSort(a, l, p-1)
	quickSort(a, p+1, r)
}

func findPos(a []int, l, r int) int {
	pos := rand.Intn(r-l+1) + l
	a[l], a[pos] = a[pos], a[l]
	p := l
	for i := l + 1; i <= r; i++ {
		if a[i] < a[l] {
			a[p+1], a[i] = a[i], a[p+1]
			p++
		}
	}
	a[p], a[l] = a[l], a[p]
	return p
}

// 堆排序
func deepDown(a []int, n, i int) {
	l, r := i<<1+1, i<<1+2
	if l >= n {
		return
	}
	if r >= n {
		if a[i] < a[l] {
			a[i], a[l] = a[l], a[i]
			deepDown(a, n, l)
		}
		return
	}
	if a[i] < a[l] {
		if a[l] < a[r] {
			a[i], a[r] = a[r], a[i]
			deepDown(a, n, r)
		} else {
			a[i], a[l] = a[l], a[i]
			deepDown(a, n, l)
		}
	} else {
		if a[i] < a[r] {
			a[i], a[r] = a[r], a[i]
			deepDown(a, n, r)
		}
	}
}
func buildHeap(a []int) {
	n := len(a)
	for i := n/2 - 1; i >= 0; i-- {
		deepDown(a, n, i)
	}
}

func heapSort(a []int) {
	buildHeap(a)
	n := len(a)
	for n > 0 {
		a[0], a[n-1] = a[n-1], a[0]
		n--
		deepDown(a, n, 0)
	}
}

// 归并排序
func mergeSort(a []int) []int {
	n := len(a)
	if n <= 1 {
		return a
	}
	left := mergeSort(a[:n/2])
	right := mergeSort(a[n/2:])
	return mergeForSort(left, right)
}
func mergeForSort(a, b []int) []int {
	m, n := len(a), len(b)
	i, j, ans := 0, 0, []int(nil)
	for i < m && j < n {
		if a[i] < b[j] {
			ans = append(ans, a[i])
			i++
		} else {
			ans = append(ans, b[j])
			j++
		}
	}
	ans = append(ans, a[i:]...)
	ans = append(ans, b[j:]...)
	return ans
}
