import java.io.*;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Created by H77 on 2017/12/5.
 */
public class pathChoose {
    HashMap<String,City> cityMaps = new HashMap<>();
    //point节点的key形式x_y，Point的数据结构看Point类
    HashMap<String,Point> points = new HashMap<>();
    //一个队列，主要提供先进先出的功能
    int moves[][] = new int[][]{{1,0},{0,1},{-1,0},{0,-1}};
    SimpleDateFormat formatter = new SimpleDateFormat("mm:ss");
    public void initCity(){
        List<String> data = CSVUtils.importCsv(new File("E:\\match\\CityData.csv"));
        for(int i = 1 ; i < data.size() ; i++) {
            String str = data.get(i);
            String[] s = str.split(",");
            City c = new City(Integer.parseInt(s[0]),Integer.parseInt(s[1]),Integer.parseInt(s[2]));
            cityMaps.put(s[0],c);
        }
    }
    //解析每个节点得到的数据是x_id,y_id,hour,wind
    public void initPoints(){
//        List<String> points = CSVUtils.importCsv(new File("E:\\比赛数据\\pointsReal.csv"));
        BufferedReader br = null;
        String max = " ";
        try {
            br = new BufferedReader(new FileReader("E:\\match\\1213\\testPoints.csv"));
            String line = "";
            br.readLine();
            while ((line = br.readLine()) != null) {
                String[] s = line.split(",");
                if("10".equals(s[0])){
                    String id = s[3]+"_"+s[4];
                    if(points.containsKey(id)) {
                        Point p = points.get(id);
                        p.putWind(Integer.parseInt(s[1]),Double.parseDouble(s[2]));
                    }else {
                        Point p = new Point();
                        p.putWind(Integer.parseInt(s[1]),Double.parseDouble(s[2]));
                        points.put(id,p);
                    }
                }else if("11".equals(s[0])){
                    break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }finally {
            if(br!=null){
                try {
                    br.close();
                    br=null;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        //436 298
        System.out.print('.');
    }
    public void output(int target,List<Position>  pos){
        File f = new File("E:\\match\\1213\\path.csv");
        int len = pos.size();
        List<String> outs = new ArrayList<>();
        for(int i = len-1 ; i > -1 ; i--){
            Position p = pos.get(i);
            int hour = p.time / 60;
            int minute = p.time % 60;
            String time = null;
            if(minute >= 10)
                time = hour+":"+minute;
            else time = hour+":0"+minute;
            if(hour <10) {
                time = "0"+time;
            }
            String res = target+","+10+","+time+","+p.x+","+p.y;
            outs.add(res);
        }
        CSVUtils.exportCsv(f,outs);
    }
    //该算法目前就考虑了一天的情况  出版代码
    public List<Position> pathChoose(int x,int y, int x1 ,int y1){
        //x y 为初始位置  x1 y1为终点位置
        //结果集

        List<Position> positions = new ArrayList<>();
        //包含了经过的路径集合，默认不会往回走,如果往回走，不如一开始就不往这个方向飞
        HashMap<String,Position> close = new HashMap<>();
        //将要被估算的候选节点
        //一部分候选节点是飞行路径中每1h记录一个
        Queue<Position> open = new LinkedList<Position>();
//        HashMap<String,Position> open = new HashMap<>();

        Queue<Position> q = new LinkedList<>();

        //起始时间是3点
        int time = 3*60;
        double windThreshold = 15;
        //初始化起始位置
        Position start = new Position(x,y,time,0);
        q.add(start);
        //count对应的是当前队列的长度，count1是下次队列加入的点的长度
        int count = 1;
        int count1 = 1;
        int up_wind = 0;

        int minTime = Integer.MAX_VALUE;

        //算法迭代终止的条件
        boolean end = false;
        int maxTimes = 100; //默认会去找20条路径
        while(!q.isEmpty()) {
            //时间限制 最多飞机飞到21点
            if(time >= 21*60){
                if(maxTimes > 0) {
                    Position p = open.poll();
                    if(p != null) {
                        time = p.time;
                        if(q.size() != 0) q.poll();
                        q.add(p);
                        maxTimes--;
                    }else{
                        break;
                    }
                }else {
                    break;
                }
            }
            time = time + 2;
            count = count1;
            count1 = 0;
            //遍历下一层节点
            while(count > 0) {
                count--;
                //得到队列中最开始的数据
                Position pre = q.poll();
//                System.out.println(pre.x+","+pre.y+","+pre.time+","+pre.wind);
                List<Position> preList = new ArrayList<>();
                up_wind = 0;
                //每个点，对于飞机来说都有4个方向可以前进
                for (int i = 0; i < 4; i++) {
                    int[] move = moves[i];
                    int x0 = pre.x + move[0];
                    int y0 = pre.y + move[1];
                    //默认飞机不能往回开
                    if (pre.pre != null && pre.pre.x == x0 && pre.pre.y == y0) continue;
                    //说明到达了目的地
                    if (x0 == x1 && y0 == y1) {
//                        List<Position> position = new ArrayList<>();
                        Position p = new Position(x0,y0,time,0);
                        p.pre = pre;
                        pre = p;
                        while (pre != null) {
                            positions.add(pre);
                            pre = pre.pre;
                        }
//                        int min = position.get(0).time - 3*60;
//                        if(min < minTime){
//                            positions = position;
//                            minTime = min;
//                        }
//                        time = 21*60;//假设超时了
//                        count1++;
                        System.out.println("reach");
                        //取局部最优解 如果有节点以及到了，直接跳出循环
                        end = true;
                        break;
                    }
                    String id = x0 + "_" + y0;
                    //说明路径可达的情况
                    if (points.containsKey(id)) {
                        double wind = 0.0;
                        //得到当前位子的风速情况
                        wind = points.get(id).getWind(time / 60);
                        if (wind >= windThreshold) {
                            //无人机会坠毁
//                            System.out.println("in");
                        } else {
                            //满足启发式条件前提下的点 暂时加入到预处理的点列表
                            int original = heuristic(pre.x,pre.y,x1,y1);
                            int after = heuristic(x0,y0,x1,y1);
                            if(after < original) {
                                Position p = new Position(x0, y0, time,wind);
                                preList.add(p);
                                p.pre = pre;
//                                q.add(p);
//                                count1++;
                            }else {
                                //wind达标 启发函数不达标的点
                                up_wind++;
                            }
                        }
                    } else {
                        //路径不可达的情况
                        System.out.print(".");
                    }
                }
                //跳出内层循环
                if(end){
                    break;
                }
                //如果preList.size等于0 说明1.飞机遇到风速>15的情况 2.启发函数不达标
                //如果有部分启发函数的点不达标，这个时候选其中风速最小的点处理
                //如果风速都>15 则倒退至前一个点重新选择
                if(preList.size() == 0 && up_wind > 0){
                    for (int i = 0; i < 4; i++) {
                        int[] move = moves[i];
                         int x0 = pre.x + move[0];
                         int y0 = pre.y + move[1];
                        if (pre.pre != null && pre.pre.x == x0 && pre.pre.y == y0) continue;
                        String id = x0 + "_" + y0;
                        if(points.containsKey(id)){
                            double wind = 0.0;
                            wind = points.get(id).getWind(time / 60);
                            if(wind < windThreshold){
                                Position p = new Position(x0, y0, time,wind);
                                preList.add(p);
                                p.pre = pre;
                            }
                        }
                    }
                }else if(preList.size() == 0 && up_wind == 0){
                    //在该种情况下可能飞机正处于暴风
                    //如果还没有起飞
                    Position e = new Position(pre.x, pre.y, time, pre.wind);
                    e.pre = pre;
                    q.add(e);
                    count1++;
//                    while (time % 60 != 0) {
//                        e = new Position(pre.x, pre.y, time, pre.wind);
//                        e.pre = pre;
//                        time = time + 2;
//                        pre = e;
//                    }
//                    if(e != null) {
//                        q.add(e);
//                        count1++;
//                    }
//                    if(pre.x == x && pre.y == y){
//                        time = time + 58;
//                        q.add(pre);
//                        count1++;
//                    }else{ //飞机在起飞的情况下周围都是风暴 如果能在原地不动 先不动一段时间
//                        time = time + 60-time % 60;
//                        q.add(pre);
//                        count1++;
//                    }
                    // 退回到上一个节点的情况
                }

                //对节点进行判断 取wind最小的节点为下个目的地
                Position min_p = null;
                Position second_min_p = null;
                double average = 0.0;
                for(int j = 0 ; j < preList.size() ; j++){
                    average += preList.get(j).wind;
                    if(min_p == null) {
                        min_p = preList.get(j);
                        continue;
                    }
                    Position p = preList.get(j);
                    double curWind = predictWind(p.pre,p);
                    double condidateWind = predictWind(min_p.pre,min_p);
                    if(curWind < condidateWind) {
                        second_min_p = min_p;
                        min_p = p;
                    }
                }
                if(second_min_p == null && preList.size() != 0) second_min_p = preList.get(preList.size()-1);
                average = average /preList.size();
                if(min_p != null) {
                    q.add(min_p);
                    //有备份情况
                    if(time % 20 == 0 && preList.size() > 1) open.add(second_min_p);
                    count1++;
                }
                //如果最小风值 与平均值偏差 大于2.0 就添加最小值
//                if((average - min_p.wind) > 0.1) {
//                    q.add(min_p);
//                    count1++;
//                }else{
//                    q.add(min_p);
//                    q.add(second_min_p);
//                    count1 = count1 + 2;
//                }
            }
            //跳出最外层循环
            if(end){
                break;
            }
        }
//        System.out.println(positions.size());
        //343
        //378
//        for(Position p : positions){
//            System.out.println(p.x+","+p.y);
//        }
        //计算飞机的飞行时间
        int t = 0;
        if(positions.size() != 0){
            t = positions.get(0).time - 3*60;
        }else {
            t = 24*60;
        }
        System.out.println("time:"+t);
        return positions;
//        System.out.println(".");
    }

    public double predictWind(Position pre , Position p ){
        int x0 = pre.x;
        int y0 = pre.y;
        int x1 = p.x;
        int y1 = p.y;
        int direct_x = x1-x0;
        int direct_y = y1-y0;
        int count = 0;
        double totalWind = 0;
        if(direct_x !=0){
            for(int i = 0 ; i < 3 ; i++){
                for(int j = -1 ; j <=1 ;j++){
                    int x = x1+direct_x;
                    int y = y1+j;
                    if(points.containsKey(x+"_"+y)){
                        count++;
                        totalWind += points.get(x+"_"+y).getWind(p.time/60);
                    }
                }
            }
        }else{
            for(int i = 0 ; i < 3 ; i++){
                for(int j = -1 ; j <=1 ;j++){
                    int x = x1+j;
                    int y = y1+direct_y;
                    if(points.containsKey(x+"_"+y)){
                        count++;
                        totalWind += points.get(x+"_"+y).getWind(p.time/60);
                    }
                }
            }
        }
        return totalWind /count;
    }

    //该算法目前就考虑了一天的情况  出版代码
    public List<Position> pathChoose1(int x,int y, int x1 ,int y1){
        //x y 为初始位置  x1 y1为终点位置
        //结果集

        List<Position> positions = new ArrayList<>();
        //包含了经过的路径集合，默认不会往回走,如果往回走，不如一开始就不往这个方向飞
        HashMap<String,Position> close = new HashMap<>();
        //将要被估算的候选节点
        //一部分候选节点是飞行路径中每1h记录一个
        Queue<Position> open = new LinkedList<Position>();
//        HashMap<String,Position> open = new HashMap<>();

        Queue<Position> q = new LinkedList<>();

        //起始时间是3点
        int time = 3*60;
        double windThreshold = 15;
        //初始化起始位置
        Position start = new Position(x,y,time,0);
        q.add(start);
        //count对应的是当前队列的长度，count1是下次队列加入的点的长度
        int count = 1;
        int count1 = 1;
        int up_wind = 0;

        int minTime = Integer.MAX_VALUE;

        //算法迭代终止的条件
        boolean end = false;
        int maxTimes = 100; //默认会去找20条路径
        while(!q.isEmpty()) {
            //时间限制 最多飞机飞到21点
            if(time >= 21*60){
                if(maxTimes > 0) {
                    Position p = open.poll();
                    if(p != null) {
                        time = p.time;
                        if(q.size() != 0) q.poll();
                        q.add(p);
                        maxTimes--;
                    }else{
                        break;
                    }
                }else {
                    break;
                }
            }
            time = time + 2;
            count = count1;
            count1 = 0;
            //遍历下一层节点
            while(count > 0) {
                count--;
                //得到队列中最开始的数据
                Position pre = q.poll();
                List<Position> preList = new ArrayList<>();
                up_wind = 0;
                //每个点，对于飞机来说都有4个方向可以前进
                for (int i = 0; i < 4; i++) {
                    int[] move = moves[i];
                    int x0 = pre.x + move[0];
                    int y0 = pre.y + move[1];
                    //默认飞机不能往回开
                    if (pre.pre != null && pre.pre.x == x0 && pre.pre.y == y0) continue;
                    //说明到达了目的地
                    if (x0 == x1 && y0 == y1) {
//                        List<Position> position = new ArrayList<>();
                        Position p = new Position(x0,y0,time,0);
                        p.pre = pre;
                        pre = p;
                        while (pre != null) {
                            positions.add(pre);
                            pre = pre.pre;
                        }
//                        int min = position.get(0).time - 3*60;
//                        if(min < minTime){
//                            positions = position;
//                            minTime = min;
//                        }
//                        time = 21*60;//假设超时了
//                        count1++;
                        System.out.println("reach");
                        //取局部最优解 如果有节点以及到了，直接跳出循环
                        end = true;
                        break;
                    }
                    String id = x0 + "_" + y0;
                    //说明路径可达的情况
                    if (points.containsKey(id)) {
                        double wind = 0.0;
                        //得到当前位子的风速情况
                        wind = points.get(id).getWind(time / 60);
                        if (wind >= windThreshold) {
                            //无人机会坠毁
//                            System.out.println("in");
                        } else {
                            //满足启发式条件前提下的点 暂时加入到预处理的点列表
                            int original = heuristic(pre.x,pre.y,x1,y1);
                            int after = heuristic(x0,y0,x1,y1);
                            if(after < original) {
                                Position p = new Position(x0, y0, time,wind);
                                preList.add(p);
                                p.pre = pre;
//                                q.add(p);
//                                count1++;
                            }else {
                                //wind达标 启发函数不达标的点
                                up_wind++;
                            }
                        }
                    } else {
                        //路径不可达的情况
                        System.out.print(".");
                    }
                }
                //跳出内层循环
                if(end){
                    break;
                }
                //如果preList.size等于0 说明1.飞机遇到风速>15的情况 2.启发函数不达标
                //如果有部分启发函数的点不达标，这个时候选其中风速最小的点处理
                //如果风速都>15 则倒退至前一个点重新选择
                if(preList.size() == 0 && up_wind > 0){
                    for (int i = 0; i < 4; i++) {
                        int[] move = moves[i];
                        int x0 = pre.x + move[0];
                        int y0 = pre.y + move[1];
                        if (pre.pre != null && pre.pre.x == x0 && pre.pre.y == y0) continue;
                        String id = x0 + "_" + y0;
                        if(points.containsKey(id)){
                            double wind = 0.0;
                            wind = points.get(id).getWind(time / 60);
                            if(wind < windThreshold){
                                Position p = new Position(x0, y0, time,wind);
                                preList.add(p);
                                p.pre = pre;
                            }
                        }
                    }
                }else if(preList.size() == 0 && up_wind == 0){
                    //在该种情况下可能飞机正处于暴风
                    //如果还没有起飞
                    Position e = new Position(pre.x, pre.y, time, pre.wind);
                    e.pre = pre;
                    q.add(e);
                    count1++;
//                    while (time % 60 != 0) {
//                        e = new Position(pre.x, pre.y, time, pre.wind);
//                        e.pre = pre;
//                        time = time + 2;
//                        pre = e;
//                    }
//                    if(e != null) {
//                        q.add(e);
//                        count1++;
//                    }
//                    if(pre.x == x && pre.y == y){
//                        time = time + 58;
//                        q.add(pre);
//                        count1++;
//                    }else{ //飞机在起飞的情况下周围都是风暴 如果能在原地不动 先不动一段时间
//                        time = time + 60-time % 60;
//                        q.add(pre);
//                        count1++;
//                    }
                    // 退回到上一个节点的情况
                }

                //对节点进行判断 取wind最小的节点为下个目的地
                Position min_p = null;
                Position second_min_p = null;
                double average = 0.0;
                for(int j = 0 ; j < preList.size() ; j++){
                    average += preList.get(j).wind;
                    if(min_p == null) {
                        min_p = preList.get(j);
                        continue;
                    }
                    if(preList.get(j).wind < min_p.wind) {
                        second_min_p = min_p;
                        min_p = preList.get(j);
                    }
                }
                if(second_min_p == null && preList.size() != 0) second_min_p = preList.get(preList.size()-1);
                average = average /preList.size();
                if(min_p != null) {
                    q.add(min_p);
                    //有备份情况
                    if(time % 20 == 0 && preList.size() > 1) open.add(second_min_p);
                    count1++;
                }
                //如果最小风值 与平均值偏差 大于2.0 就添加最小值
//                if((average - min_p.wind) > 0.1) {
//                    q.add(min_p);
//                    count1++;
//                }else{
//                    q.add(min_p);
//                    q.add(second_min_p);
//                    count1 = count1 + 2;
//                }
            }
            //跳出最外层循环
            if(end){
                break;
            }
        }
//        System.out.println(positions.size());
        //343
        //378
//        for(Position p : positions){
//            System.out.println(p.x+","+p.y);
//        }
        //计算飞机的飞行时间
        int t = 0;
        if(positions.size() != 0){
            t = positions.get(0).time - 3*60;
        }else {
            t = 24*60;
        }
        System.out.println("time:"+t);
        return positions;
//        System.out.println(".");
    }


    public void createCSV(){
        File f = new File("E:\\match\\1213\\path.csv");
        if(!f.exists()) {
            try {
                f.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        List<String> lists = new ArrayList<String>();
        String head = "目的地编号,日期编号,时间,x轴坐标,y轴坐标";
        lists.add(head);
        CSVUtils.exportCsv(f,lists);
    }


    public void computeTime(){
         int time = 0;
         initCity();
         initPoints();
         for(int i = 1 ; i <= 10 ; i ++){
             City c = cityMaps.get(i+"");
             List<Position>  p = pathChoose(142,328,c.getX(),c.getY());
             if(p.size() != 0) {
                 output(c.getId(),p);
             }
         }
        System.out.println(time);
    }

    public int heuristic(int x, int y ,int target_x ,int target_y){
        int dx = Math.abs(target_x-x);
        int dy = Math.abs(target_y-y);
        return dx+dy;
    }

    public static void main(String[] args) {
          pathChoose p = new pathChoose();
//        p.createCSV();
          p.computeTime();
//        p.initPoints();
//        List<Position> positions =p.pathChoose(142,328,236,241);
//        p.output(2,positions);
    }
}
