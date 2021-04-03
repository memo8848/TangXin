import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:test1/home/Home.dart';
import 'package:test1/models/User.dart';
import 'package:dio/dio.dart';
import 'package:flutter/services.dart';
import 'dart:convert';
import 'dart:async';
import '../models/User.dart';
import 'package:test1/theme.dart' as theme;

class LoginScreen extends StatefulWidget {
  @override
  State<LoginScreen> createState() {
    return new _LoginScreenState();
  }
}

class _LoginScreenState extends State<LoginScreen> {
  final _formKey = GlobalKey<FormState>();

  //参数 账号密码
  String _user, _password;
  //是否显示密码
  bool _isObscure = true;
  Color _eyeColor;
  //请求结果 也就是个人信息，要传给主页，也要存储
  var _result = '';
  var _decodeResult = '';
  var data;
  //登录请求函数
  loadDataByDio() async {
    try {
      print('登陆中');
      Response response;
      Dio dio = new Dio();
      response = await dio.post("http://10.8.51.45:3050/api/pc/v1/users/login",
          data: {"email": _user, "password": _password});
      if (response.statusCode == 200) {
        new MaterialPageRoute(builder: (context) => new Home());
        _decodeTest(response.data);
      } else {
        _result = 'error code : ${response.statusCode}';
      }
    } catch (exception) {
      print('exc:$exception');
      _result = '网络异常';
    }
    setState(() {});
  }

//解析信息
  _decodeTest(var body) {
    bool usertxt = body['success'];
    String errMess = body['message'];

    if (usertxt == false) {
      _modalBottomSheetMenu(errMess);
    } else {
      print('body 类型是  ${body.runtimeType}');
      print(body);
      User user = User.fromJson(body['data']);
      _RemenberUser(user);
      print(
          'person name is ${user.name}, age is ${user.email}, height is ${user.title}');
      Navigator.of(context).pushAndRemoveUntil(
          new MaterialPageRoute(builder: (context) => new Home()),
          (route) => route == null);
      _testAnalyse();
    }
  }

  _testAnalyse() {
    Future<String> _loadAStudentAsset() async {
      return await rootBundle.loadString('jsons/user.json');
    }

    Future loadStudent() async {
      String jsonString = await _loadAStudentAsset();
      final jsonResponse = json.decode(jsonString);
      User student = new User.fromJson(jsonResponse);
      print(student.entry_year);
    }
  }

  _RemenberUser(User user) async {

    print('输出信息 ${user.name}, 邮箱 is ${user.email}, height is ${user.title}');
    SharedPreferences sharedPreferences = await SharedPreferences.getInstance();
    sharedPreferences.setStringList("class_id", user.class_id);
    sharedPreferences.setString("email", user.email);
    sharedPreferences.setString("title", user.title);
    sharedPreferences.setString("photo", user.photo);
    sharedPreferences.setString("role", user.role);
    sharedPreferences.setString("entry_year", user.entry_year);
    sharedPreferences.setString("id", user.id);
    sharedPreferences.setString("name", user.name);
    sharedPreferences.setString("user_id", user.user_id);
    sharedPreferences.setString("org_name", user.org_name);
    sharedPreferences.setString("subOrg_name", user.subOrg_name);
    sharedPreferences.setString("major_name", user.major_name);
    sharedPreferences.setStringList("resources", user.resources);
    sharedPreferences.setString("org_id", user.org_id);

    print('存入信息'+sharedPreferences.getString("email"));

 
  }

  _modalBottomSheetMenu(errMess) {
    showModalBottomSheet(
        context: context,
        builder: (builder) {
          return new Container(
            height: 20.0,
            child: new Container(
                decoration: new BoxDecoration(
                  color: Colors.grey,
                ),
                child: new Center(
                  child: new Text(
                    errMess,
                    style: TextStyle(color: Colors.black),
                  ),
                )),
          );
        });
  }

//账号输入框
  TextFormField buildUserTextField() {
    return TextFormField(
      validator: (String value) {
        if (value.isEmpty) {
          return '请输入邮箱';
        }
      },
      style: TextStyle(color: theme.Theme.black, fontSize: 18.0),
      cursorColor: theme.Theme.moss,
      decoration: InputDecoration(
          fillColor: theme.Theme.background,
          filled: true,
          contentPadding: EdgeInsets.only(top: 13),
          prefixIcon: Icon(Icons.person),
          hintText: 'name@snnu.edu.cn',
          hintStyle: TextStyle(color: theme.Theme.black, fontSize: 18.0,fontWeight: FontWeight.w300),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(21.11), //边框裁剪成11.11°角
            borderSide: BorderSide(
                color: Colors.black, width: 30.0), //没有效果，想要修改就返回Theme（如前面账号样式）
          )),
      onSaved: (String value) => _user = value,
    );
  }

//密码输入框
  TextFormField buildPasswordTextField(BuildContext context) {
    return TextFormField(
      onSaved: (String value) => _password = value,
      obscureText: _isObscure,
      validator: (String value) {
        if (value.isEmpty) {
          return '请输入密码';
        }
      },
      style: TextStyle(color: theme.Theme.black, fontSize: 18.0),
      cursorColor: theme.Theme.black,
      decoration: InputDecoration(
          fillColor: theme.Theme.background,
          filled: true,
          contentPadding: EdgeInsets.only(top: 13),
          prefixIcon: Icon(Icons.lock),
          hintText: 'password',
          hintStyle: TextStyle(color: theme.Theme.black, fontSize: 18.0,fontWeight: FontWeight.w300),
          suffixIcon: IconButton(
              icon: Icon(
                Icons.remove_red_eye,
                color: _eyeColor,
              ),
              onPressed: () {
                setState(() {
                  _isObscure = !_isObscure;
                  _eyeColor = _isObscure
                      ? Color.fromARGB(255, 166, 166, 166)
                      : Color.fromARGB(255, 41, 41, 41);
                });
              }),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(21.11), //边框裁剪成11.11°角
            borderSide: BorderSide(
                color: Colors.black, width: 30.0), //没有效果，想要修改就返回Theme（如前面账号样式）
          )),
    );
  }

//登录按钮
  Align buildLoginButton(BuildContext context) {
    return Align(
      child: SizedBox(
        height: 45.0,
        width: 250.0,
        child: RaisedButton(
          child: Text(
            'LOGIN',
            style: TextStyle(color: Colors.white, fontSize: 20.0),
          ),
          color: theme.Theme.moss,
          onPressed: () {
            if (_formKey.currentState.validate()) {
              //只有输入的内容符合要求通过才会到达此处
              _formKey.currentState.save();
              loadDataByDio();
            }
          },
          shape: StadiumBorder(side: BorderSide.none),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: theme.Theme.green,
      appBar: new AppBar(
        title: new Text('登录'),
        backgroundColor: theme.Theme.moss,
      ),
      body: new SafeArea(
          child: new SingleChildScrollView(
              child: new Container(
                  height: MediaQuery.of(context).size.height,
                  width: MediaQuery.of(context).size.width,
                  
                  child: new Column(
                      mainAxisSize: MainAxisSize.max,
                      
                      children: <Widget>[
                         SizedBox(
                          height: 60,
                        ),
                        new Container(
                            child:
                                CircleAvatar(
                                radius: 80,
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
                          height: 30,
                        ),
                        Form(
                            key: _formKey,
                            child: 
                             Container(
                                height: 330,
                                width: 300,
                                alignment: Alignment.center,
                                decoration: BoxDecoration(
                                    //背景装饰
                                    color: theme.Theme.dust,
                                     borderRadius: BorderRadius.circular(11.21), //边框裁剪成11.11°角
                                    boxShadow: [
                                      //卡片阴影
                                      BoxShadow(
                                        
                                          color: Colors.black54,
                                          offset: Offset(2.0, 2.0),
                                          blurRadius: 4.0)
                                    ]), 
                           
                          
                                  
                           child:
                                  ListView(
                                    padding: EdgeInsets.symmetric(
                                        horizontal: 20.0, vertical: 20.0),
                                    children: <Widget>[
                                      Container(height: 40.0,
                                      child: Text(
                                        'LOGIN',
                                       style: TextStyle(
                                        fontSize: 35.0,
                                        color: theme.Theme.black,
                                        fontWeight: FontWeight.bold,
                                      ),
                                      ),
                                      ),
                                       Container(height: 20.0,
                                
                                      child: Text(
                                        'account number is name',
                                       style: TextStyle(
                                        fontSize: 14.0,
                                        color: theme.Theme.black,
                                        fontWeight: FontWeight.w300,
                                      ),
                                      ),
                                      ),
  Divider(
                color: theme.Theme.black,
                
              ),
                                      SizedBox(height: 20.0),
                                      
                                      //账号
                                      buildUserTextField(),

                                      SizedBox(height: 15.0),
                                      //密码
                                      buildPasswordTextField(context),

                                    /*添加记住密码的按钮 flatbutton */

                                      SizedBox(height: 20.0),
                                      //登录按钮
                                      buildLoginButton(context),
                                    ]
                                    )
                                )
                                    )
                        ])))),
    );
  }

  text() {}
}
