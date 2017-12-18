import java.io.BufferedReader;
import java.io.FileReader;

/**
 * Created by H77 on 2017/12/16.
 */
public class countNum {

    public int[] arr = new int[50];

    public void dealData(){
        BufferedReader br = null;
        String max = " ";
        try {
            br = new BufferedReader(new FileReader("E:\\match\\1213\\1.txt"));
            String line = "";
            int index = 0;
            int count = 0;
            while ((line = br.readLine()) != null) {
                if(!line.contains(":")) break;
                if(Integer.parseInt(line.split(":")[1]) == 1440){
                    count++;
                    continue;
                }else {
                    arr[index] = Integer.parseInt(line.split(":")[1]);
                    index++;
                }
            }


            int len = index+1;
            int sum = 63176 - count * 1440;
            //2796 8
            // 63176
            int s = 0;
            for(int i = 0 ; i < len ; i++){
                s = s + arr[i];
            }
            // 208 188 322 128 364 648 294 622 394 412 128 742 966 376
            // 376 966   208 188 648 128 128  364 294 394
            //           322 622 412 742
            int ave = s / len;
            System.out.println(ave);
        }catch (Exception e){
           e.printStackTrace();
        }


    }
    public static void main(String[] args){
        countNum c = new countNum();
        c.dealData();
    }
}
