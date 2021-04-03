
class User {
  String email;
  List<String> class_id;
  String title; //学生
  String photo;
  String role; //student
  String entry_year;
  String id;
  String name;
  String user_id; //学号
  String org_name; //学校名
  String subOrg_name; //计算机科学学院
  String major_name; //软件工程
  List<String> resources;
  String org_id; //6048c1c7c9ffe60020d149f6
  //构造函数 只有这俩 就是一个学生对象
  
  User({this.email,
        this.class_id,
        this.title,
        this.photo,
        this.role,
        this.entry_year,
        this.id,
        this.name,
        this.user_id,
        this.org_name,
        this.subOrg_name,
        this.major_name,
        this.resources,
        this.org_id,});

  factory User.fromJson(Map<String, dynamic> json) {

    var classList = json['class_id'];
    List<String> class_list  = new List<String>.from(classList);

    var resourcesList = json['class_id'];
    List<String> resources_list  = new List<String>.from(resourcesList);


    return User(email: json['emial'],
     class_id: class_list, 
     title: json['title'],
     photo: json['photo'],
     role: json['role'],
     entry_year: json['entry_year'],
     id: json['_id'],
     name: json['name'],
     user_id: json['user_id'],
     org_name: json['org_name'],
     subOrg_name: json['subOrg_name'],
    major_name: json['major_name'],
    resources:resources_list,
    org_id: json['org_id'],);
  }


}
