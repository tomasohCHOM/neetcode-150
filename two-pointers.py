from typing import List


class Solution:
    # Valid Palindrome (Easy)
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        while l < r:
            while l < r and not s[l].isalnum():
                l += 1
            while l < r and not s[r].isalnum():
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l, r = l + 1, r - 1
        return True

    # Two Sum II - Input Array Is Sorted (Medium)
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers) - 1
        while l < r:
            num_sum = numbers[l] + numbers[r]
            if num_sum > target:
                r -= 1
            elif num_sum < target:
                l += 1
            else:
                return [l + 1, r + 1]

    # 3Sum (Medium)
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        output = []
        for i in range(len(nums) - 2):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            j, k = i + 1, len(nums) - 1
            while j < k:
                num_sum = nums[i] + nums[j] + nums[k]
                if num_sum < 0:
                    j += 1
                elif num_sum > 0:
                    k -= 1
                else:
                    output.append((nums[i], nums[j], nums[k]))
                    while j < k and nums[j] == nums[j + 1]:
                        j += 1
                    while j < k and nums[k] == nums[k - 1]:
                        k -= 1
                    j += 1
                    k -= 1
        return output

    # Container With Most Water (Medium)
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        max_h = max(height)
        output = 0
        while l < r:
            area = (r - l) * min(height[l], height[r])
            output = max(output, area)
            if max_h * (r - l) <= output:
                break
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return output

    # Trapping Rain Water (Hard)
    def trap(self, height: List[int]) -> int:
        # Why does it work? Because by updating the pointer from the side (left,
        # right) with a lesser maximum height, we ensure that we accumulate the
        # most amount of water that we can get. If both are equal, it does not
        # matter which one we update.
        l, r = 0, len(height) - 1
        output = 0
        max_l, max_r = height[l], height[r]
        while l < r:
            if max_l < max_r:
                l += 1
                max_l = max(max_l, height[l])
                output += max_l - height[l]
            else:
                r -= 1
                max_r = max(max_r, height[r])
                output += max_r - height[r]
        return output
