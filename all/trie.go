package all

// 字典树
type Trie struct {
	children [26]*Trie
	isEnd    bool
}

func (t *Trie) Insert(word string) {
	trav := t
	for _, v := range word {
		ch := int(v - 'a')
		if trav.children[ch] == nil {
			trav.children[ch] = &Trie{}
		}
		trav = trav.children[ch]
	}
	trav.isEnd = true
}

func (t *Trie) Search(word string) bool {
	trav := t
	for _, v := range word {
		ch := int(v - 'a')
		if trav.children[ch] == nil {
			return false
		}
		trav = trav.children[ch]
	}
	return trav.isEnd
}

func (t *Trie) StartsWith(prefix string) bool {
	trav := t
	for _, v := range prefix {
		ch := int(v - 'a')
		if trav.children[ch] == nil {
			return false
		}
		trav = trav.children[ch]
	}
	return true
}

// type Trie struct {
// 	children [26]*Trie
// 	word     string // 或者为isEnd bool
// }

// func (t *Trie) insert(word string) {
// 	trav := t
// 	for _, v := range word {
// 		ch := int(v - 'a')
// 		if trav.children[ch] == nil {
// 			trav.children[ch] = &Trie{}
// 		}
// 		trav = trav.children[ch]
// 	}
// 	trav.word = word
// }

// 单词搜索II   https://leetcode.cn/problems/word-search-ii/   字典树 + 回溯

// 统计异或值在范围内的数对有多少    https://leetcode.cn/problems/count-pairs-with-xor-in-a-range/      二进制字典树
// 求解有多少对数字的异或运算结果处于 [low,high] 之间，为了方便求解，我们用 f(x) 来表示有多少对数字的异或运算结果小于等于 x，
// 这时问题变为求解 f(high)−f(low−1)
