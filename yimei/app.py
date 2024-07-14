from flask import Flask, render_template, request, redirect, url_for  
import pymysql  
from reportlab.pdfgen import canvas  
from reportlab.lib.pagesizes import letter  

 
  
# 数据库配置（请根据实际情况修改）  
db_config = {  
    'host': 'localhost',  
    'user': 'root',  
    'password': 'password',  
    'db': 'medical_beauty'  
}  
  
# 连接数据库  
def get_db_connection():  
    return pymysql.connect(**db_config)  
  
# 产品展示页面  
@app.route('/')  
def index():  
    conn = get_db_connection()  
    cursor = conn.cursor()  
    cursor.execute("SELECT * FROM products")  
    products = cursor.fetchall()  
    cursor.close()  
    conn.close()  
    return render_template('index.html', products=products)  
  
# 会员注册页面  
@app.route('/register', methods=['GET', 'POST'])  
def register():  
    if request.method == 'POST':  
        username = request.form['username']  
        password = request.form['password']  
        level = request.form['level']  
        conn = get_db_connection()  
        cursor = conn.cursor()  
        cursor.execute("INSERT INTO members (username, password, level) VALUES (%s, %s, %s)", (username, password, level))  
        conn.commit()  
        cursor.close()  
        conn.close()  
        return redirect(url_for('index'))  
    return render_template('register.html')  
  
# 打印诊疗单  
@app.route('/print_invoice')  
def print_invoice():  
    # 这里应该从数据库中获取客户信息和诊疗项目，然后生成PDF  
    # 为了简化，这里只创建一个空白的PDF文件  
    pdf_filename = 'invoice.pdf'  
    c = canvas.Canvas(pdf_filename, pagesize=letter)  
    c.drawString(100, 100, "诊疗单")  
    c.save()  
    return redirect(url_for('static', filename='invoice.pdf'))  
  
if __name__ == '__main__':  
    app.run(debug=True)  