import Content from '../components/Content';
import { LeetCode } from "leetcode-query";
export default async function Page() {
  const leetcode = new LeetCode();
  // const problem = await leetcode.problem("add-two-numbers");
  const problems = await leetcode.problems({
    filters: {
      difficulty: "HARD",
    }
  });
  const data = problems.questions;
  console.log(data)

  return <Content data={data} />;
}
