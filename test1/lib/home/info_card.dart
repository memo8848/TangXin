import 'package:flutter/material.dart';
import 'package:flutter/cupertino.dart';
import 'package:test1/theme.dart' as theme;

class InfoCard extends StatelessWidget {
  InfoCard({@required this.text, @required this.icon});

  final String text;
  final IconData icon;

  Widget build(BuildContext context) {
    return GestureDetector(
        child: Card(
          
            color: theme.Theme.dust,
            margin: EdgeInsets.symmetric(vertical: 5, horizontal: 25),
            shape: const RoundedRectangleBorder(//形状
          //修改圆角
          borderRadius: BorderRadius.all(Radius.circular(20)),
        ),
            child: ListTile(
              leading: Icon(
                icon,
                color: theme.Theme.black,
              ),
              title: Text(
                text,
                style: TextStyle(
                    color: theme.Theme.black,
                    fontSize: 18,
                    fontWeight: FontWeight.w300),
              ),
             
            )));
  }
}
