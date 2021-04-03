import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:test1/home/info_card.dart';
import 'package:test1/theme.dart' as theme;

class Profile extends StatefulWidget {
  @override
  State<Profile> createState() {
    return new _ProfileState();
  }
}

class _ProfileState extends State<Profile> {
  SharedPreferences sharedPreferences;
  String entry_year;
  String title;
  String role;
  String name;
  String user_id;
  String org_name;
  String subOrg_name;
  String major_name;

  initContent() async {
     SharedPreferences sharedPreferences = await SharedPreferences.getInstance();
    setState(() {  
    this.entry_year = sharedPreferences.getString("entry_year");
    this.title = sharedPreferences.getString("title");
    this.role = sharedPreferences.getString("role");
    this.name = sharedPreferences.getString("name");
    this.user_id = sharedPreferences.getString("user_id");
    this.org_name = sharedPreferences.getString("org_name");
    this.subOrg_name = sharedPreferences.getString("subOrg_name");
    this.major_name = sharedPreferences.getString("major_name");
          
        });
    

    // print('在个人主页取出信息' + sharedPreferences.getString(""));
    print('在个人主页赋值' + this.entry_year);
  }

  @override
  Widget build(BuildContext context) {
    initContent();
    return Scaffold(
      backgroundColor: theme.Theme.green,
      body: SafeArea(
        
        minimum: const EdgeInsets.only(top: 40),
        child: Column(
          children: <Widget>[
            new Container(
                            child:
                                CircleAvatar(
                                radius: 60,
                                backgroundImage: AssetImage('images/avatar.jpg'),

                              ),
                  decoration: BoxDecoration(
                                    //背景装饰
                                    borderRadius: BorderRadius.circular(90), //边框裁剪成11.11°角
                                    color: theme.Theme.dust,
                                   
                                 boxShadow: [
                                      //卡片阴影
                                      BoxShadow(
                                          color: Colors.black54,
                                          offset: Offset(2.0, 2.0),
                                          blurRadius: 4.0)
                                    ]), 
                        ),
                        SizedBox(
                          height: 25,
                        ),
                        
            Text(
              '$name',
              style: TextStyle(
                fontSize: 30.0,
                color: theme.Theme.black,
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(
              '$title',
              style: TextStyle(
                fontSize: 20.0,
                color: theme.Theme.black,
                letterSpacing: 2.5,
                fontWeight: FontWeight.w300,
              ),
            ),
            SizedBox(
              height: 20,
              width: 200,
              child: Divider(
                color: theme.Theme.black,
                indent: 10,
              ),
            ),
            InfoCard(text: '学校：${org_name}', icon: Icons.location_city),
            InfoCard(text: '学号：${this.user_id}' , icon: Icons.person),
            InfoCard(text: '学院：${this.subOrg_name}', icon: Icons.domain),
            InfoCard(text: '专业：${this.major_name}' , icon: Icons.school),
            InfoCard(text: '入学时间：${this.entry_year}' , icon: Icons.book),
          ],
        ),
      ),
    );
  }
}
