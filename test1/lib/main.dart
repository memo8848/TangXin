import 'package:flutter/material.dart';
import 'package:test1/home/Home.dart';
import 'package:test1/login/LoginScreen.dart';
import 'package:test1/theme.dart' as theme;


Map<String, WidgetBuilder> routes;


void main() =>  runApp(MyApp());
class MyApp extends StatelessWidget {
  
  @override
  Widget build(BuildContext context) {
    return new MaterialApp(
        title: 'Flutter Demo',
        theme: new ThemeData(
          primarySwatch: Colors.green
        ),
        routes: {
          /**
         * 命名导航路由，启动程序默认打开的是以'/'对应的界面LoginScreen()
         * 凡是后面使用Navigator.of(context).pushNamed('/Home')，都会跳转到Home()，
         */
        '/': (BuildContext context) => new Home(),
          // '/': (BuildContext context) => new LoginScreen(),
          '/Home': (BuildContext context) => new Home(),
        }
        );
  }
  
}
