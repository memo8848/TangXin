import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';


class MineScreen extends StatelessWidget {
  const MineScreen({ Key key }) : super(key: key);
  Widget build(BuildContext context) {
    return new Scaffold(
     body: new Center(
        child: new RaisedButton(
            child: new Text('退出登录'),
            onPressed: () {
              Navigator.of(context)
                  .pushNamed('/'); //跳转到main.dart对routeName（'/'）的界面
            }),
      ),
     
    );
  }
}