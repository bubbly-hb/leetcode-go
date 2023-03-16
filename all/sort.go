package all

import (
	"math/rand"
)

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
