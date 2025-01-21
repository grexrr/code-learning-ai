def test_environment():
    print("Testing mapper with sample input...")
    
    # test input
    sample_text = """Hello World
    Hello MapReduce
    This is a test"""


    with open('sample_input.txt', 'w') as f:
        f.write(sample_text)
    
    print("Sample input file created. You can now test the mapper using:")
    print("cat sample_input.txt | python mapper.py")

if __name__ == "__main__":
    test_environment()

# 1
class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        if not head:
            return None
        if not head.next:
            return head
        
        mid = self._getMid(head)
        root = TreeNode(mid.val)

        root.right = self.sortedListToBST(mid.next)
        mid.next = None
        root.left = self.sortedListToBST(head)
        return root

    def _getMid(self, head):
        s = head
        f = head
        prev = None
        while f and f.next:
            s = s.next
            prev = s
            f = f.next.next
        if prev:
            prev.next = None
        return s
    
# 2
class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        if not head:
            return None
        if not head.next:
            return TreeNode(head.val)
        
        mid = self._getMid(head)
        root = TreeNode(mid.val)
        root.right = self.sortedListToBST(mid.next)
        mid.next = None
        root.left = self.sortedListToBST(head)
        return root
    
    def _getMid(self, head: ListNode) -> ListNode:
        fast = head
        slow = head
        prev = None
        while fast and fast.next:
            fast = fast.next.next
            prev = slow
            slow = slow.next
        if prev:
            prev.next = None
        return slow