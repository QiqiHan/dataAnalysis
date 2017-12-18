import java.util.HashMap;

/**
 * Created by H77 on 2017/12/5.
 */
public class Point {
//    private int x;
//    private int y;
    //每个hour对应一个wind
    double[] winds = new double[30];

    public void putWind(int hour, double wind){
        winds[hour] = wind;
    }
    public  double getWind(int hour){
        return winds[hour];
    }

}
