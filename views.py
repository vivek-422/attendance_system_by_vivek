from django.shortcuts import render, redirect
from .forms import usernameForm, DateForm, UsernameAndDateForm, DateForm_2
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import time
from attendance_system_facial_recognition.settings import BASE_DIR
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from django_pandas.io import read_frame
from users.models import Present, Time
import seaborn as sns
import pandas as pd
from django.db.models import Count
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math
from django.http import HttpResponse
import csv

mpl.use('Agg')

#utility functions:
def username_present(username):
	if User.objects.filter(username=username).exists():
		return True
	
	return False

def create_dataset(username):
	id = username
	if not os.path.exists(f'face_recognition_data/training_dataset/{id}/'):
		os.makedirs(f'face_recognition_data/training_dataset/{id}/')
	directory = f'face_recognition_data/training_dataset/{id}/'

	print("[INFO] Loading the facial detector")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
	fa = FaceAligner(predictor, desiredFaceWidth=96)

	print("[INFO] Initializing Video stream")
	vs = VideoStream(src=0).start()
	sampleNum = 0
	original_images = []

	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=800)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray_frame, 0)

		for face in faces:
			print("inside for loop")
			(x, y, w, h) = face_utils.rect_to_bb(face)
			face_aligned = fa.align(frame, gray_frame, face)
			sampleNum += 1

			if face is None:
				print("face is none")
				continue

			cv2.imwrite(f'{directory}/{sampleNum}.jpg', face_aligned)
			original_images.append(face_aligned)
			face_aligned = imutils.resize(face_aligned, width=400)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
			cv2.waitKey(50)

		cv2.imshow("Add Images", frame)
		cv2.waitKey(1)
		if sampleNum > 100:
			break

	vs.stop()
	cv2.destroyAllWindows()

	# Augment to 300 images
	augmented_count = 0
	while sampleNum + augmented_count < 300:
		for img in original_images:
			if sampleNum + augmented_count >= 300:
				break
			# Flip horizontally
			flipped = cv2.flip(img, 1)
			sampleNum += 1
			augmented_count += 1
			cv2.imwrite(f'{directory}/{sampleNum}.jpg', flipped)
			
			# Rotate 5 degrees
			rows, cols = img.shape[:2]
			M = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)
			rotated = cv2.warpAffine(img, M, (cols, rows))
			sampleNum += 1
			augmented_count += 1
			cv2.imwrite(f'{directory}/{sampleNum}.jpg', rotated)

def predict(face_aligned, svc, threshold=0.6):
	face_encodings = np.zeros((1, 128))
	try:
		x_face_locations = face_recognition.face_locations(face_aligned)
		faces_encodings = face_recognition.face_encodings(face_aligned, known_face_locations=x_face_locations)
		if len(faces_encodings) == 0:
			return ([-1], [0])
	except:
		return ([-1], [0])

	prob = svc.predict_proba(faces_encodings)
	result = np.where(prob[0] == np.amax(prob[0]))
	if prob[0][result[0]] <= threshold:
		return ([-1], prob[0][result[0]])
	return (result[0], prob[0][result[0]])


def vizualize_Data(embedded, targets):
	X_embedded = TSNE(n_components=2).fit_transform(embedded)
	for i, t in enumerate(set(targets)):
		idx = targets == t
		plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)
	plt.legend(bbox_to_anchor=(1, 1))
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
	plt.close()


def update_attendance_in_db_in(present):
	today = datetime.date.today()
	time = datetime.datetime.now()
	for person in present:
		user = User.objects.get(username=person)
		try:
			qs = Present.objects.get(user=user, date=today)
		except:
			qs = None
		
		if qs is None:
			if present[person]:
				a = Present(user=user, date=today, present=True)
				a.save()
			else:
				a = Present(user=user, date=today, present=False)
				a.save()
		else:
			if present[person]:
				qs.present = True
				qs.save(update_fields=['present'])
		if present[person]:
			a = Time(user=user, date=today, time=time, out=False)
			a.save()


def update_attendance_in_db_out(present):
	today = datetime.date.today()
	time = datetime.datetime.now()
	for person in present:
		user = User.objects.get(username=person)
		if present[person]:
			a = Time(user=user, date=today, time=time, out=True)
			a.save()


def check_validity_times(times_all):
	if len(times_all) > 0:
		sign = times_all.first().out
	else:
		sign = True
	times_in = times_all.filter(out=False)
	times_out = times_all.filter(out=True)
	if len(times_in) != len(times_out):
		sign = True
	break_hourss = 0
	if sign:
		check = False
		break_hourss = 0
		return (check, break_hourss)
	prev = True
	prev_time = times_all.first().time
	for obj in times_all:
		curr = obj.out
		if curr == prev:
			check = False
			break_hourss = 0
			return (check, break_hourss)
		if curr == False:
			curr_time = obj.time
			to = curr_time
			ti = prev_time
			break_time = ((to - ti).total_seconds()) / 3600
			break_hourss += break_time
		else:
			prev_time = obj.time
		prev = curr
	return (True, break_hourss)


def convert_hours_to_hours_mins(hours):
	h = int(hours)
	hours -= h
	m = hours * 60
	m = math.ceil(m)
	return f"{h} hrs {m} mins"


def hours_vs_date_given_employee(present_qs, time_qs, admin=True):
	register_matplotlib_converters()
	df_hours = []
	df_break_hours = []
	qs = present_qs

	for obj in qs:
		date = obj.date
		times_in = time_qs.filter(date=date).filter(out=False).order_by('time')
		times_out = time_qs.filter(date=date).filter(out=True).order_by('time')
		times_all = time_qs.filter(date=date).order_by('time')
		obj.time_in = None
		obj.time_out = None
		obj.hours = 0
		obj.break_hours = 0
		if len(times_in) > 0:
			obj.time_in = times_in.first().time
		
		if len(times_out) > 0:
			obj.time_out = times_out.last().time

		if obj.time_in is not None and obj.time_out is not None:
			ti = obj.time_in
			to = obj.time_out
			hours = ((to - ti).total_seconds()) / 3600
			obj.hours = hours
		

		else:
			obj.hours = 0

		(check, break_hourss) = check_validity_times(times_all)
		if check:
			obj.break_hours = break_hourss


		else:
			obj.break_hours = 0


		
		df_hours.append(obj.hours)
		df_break_hours.append(obj.break_hours)
		obj.hours_display = convert_hours_to_hours_mins(obj.hours)
		obj.break_hours_display = convert_hours_to_hours_mins(obj.break_hours)
		obj.numeric_hours = obj.hours  # Store numeric hours for calculations
			
	
	
	
	df = read_frame(qs)	
	
	
	df["hours"] = df_hours
	df["break_hours"] = df_break_hours

	print(df)
	
	if not df.empty:
		sns.barplot(data=df, x='date', y='hours')
		plt.xticks(rotation='vertical')
		rcParams.update({'figure.autolayout': True})
		plt.tight_layout()
		if(admin):
			plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_date/1.png')
			plt.close()
		else:
			plt.savefig('./recognition/static/recognition/img/attendance_graphs/employee_login/1.png')
			plt.close()
	return qs
	

#used
def hours_vs_employee_given_date(present_qs, time_qs):
	register_matplotlib_converters()
	df_hours = []
	df_break_hours = []
	df_username = []
	qs = present_qs

	for obj in qs:
		user = obj.user
		times_in = time_qs.filter(user=user).filter(out=False)
		times_out = time_qs.filter(user=user).filter(out=True)
		times_all = time_qs.filter(user=user)
		obj.time_in = None
		obj.time_out = None
		obj.hours = 0
		obj.break_hours = 0
		if len(times_in) > 0:
			obj.time_in = times_in.first().time
		if len(times_out) > 0:
			obj.time_out = times_out.last().time
		if obj.time_in is not None and obj.time_out is not None:
			ti = obj.time_in
			to = obj.time_out
			hours = ((to - ti).total_seconds()) / 3600
			obj.hours = hours
		else:
			obj.hours = 0
		(check, break_hourss) = check_validity_times(times_all)
		if check:
			obj.break_hours = break_hourss
		else:
			obj.break_hours = 0

		df_hours.append(obj.hours)
		df_username.append(user.username)
		df_break_hours.append(obj.break_hours)
		obj.hours_display = convert_hours_to_hours_mins(obj.hours)
		obj.break_hours_display = convert_hours_to_hours_mins(obj.break_hours)
		obj.numeric_hours = obj.hours  # Store numeric hours for calculations

	df = read_frame(qs)
	df['hours'] = df_hours
	df['username'] = df_username
	df["break_hours"] = df_break_hours

	if not df.empty:
		sns.barplot(data=df, x='username', y='hours')
		plt.xticks(rotation='vertical')
		rcParams.update({'figure.autolayout': True})
		plt.tight_layout()
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_employee/1.png')
		plt.close()
	return qs


def total_number_employees():
	qs = User.objects.all()
	return len(qs) - 1


def employees_present_today():
	today = datetime.date.today()
	qs = Present.objects.filter(date=today).filter(present=True)
	return len(qs)


def this_week_emp_count_vs_date():
	today = datetime.date.today()
	some_day_last_week = today - datetime.timedelta(days=7)
	monday_of_last_week = some_day_last_week - datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs = Present.objects.filter(date__gte=monday_of_this_week).filter(date__lte=today)
	str_dates = []
	emp_count = []
	str_dates_all = []
	emp_cnt_all = []
	cnt = 0

	for obj in qs:
		date = obj.date
		str_dates.append(str(date))
		qs = Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))

	while cnt < 5:
		date = str(monday_of_this_week + datetime.timedelta(days=cnt))
		cnt += 1
		str_dates_all.append(date)
		if str_dates.count(date) > 0:
			idx = str_dates.index(date)
			emp_cnt_all.append(emp_count[idx])
		else:
			emp_cnt_all.append(0)

	df = pd.DataFrame()
	df["date"] = str_dates_all
	df["Number of employees"] = emp_cnt_all
	
	sns.lineplot(data=df, x='date', y='Number of employees')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/this_week/1.png')
	plt.close()


def last_week_emp_count_vs_date():
	today = datetime.date.today()
	some_day_last_week = today - datetime.timedelta(days=7)
	monday_of_last_week = some_day_last_week - datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs = Present.objects.filter(date__gte=monday_of_last_week).filter(date__lt=monday_of_this_week)
	str_dates = []
	emp_count = []


	str_dates_all = []
	emp_cnt_all = []
	cnt = 0

	for obj in qs:
		date = obj.date
		str_dates.append(str(date))
		qs = Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))

	while cnt < 5:
		date = str(monday_of_last_week + datetime.timedelta(days=cnt))
		cnt += 1
		str_dates_all.append(date)
		if str_dates.count(date) > 0:
			idx = str_dates.index(date)
			emp_cnt_all.append(emp_count[idx])
		else:
			emp_cnt_all.append(0)

	df = pd.DataFrame()
	df["date"] = str_dates_all
	df["emp_count"] = emp_cnt_all

	sns.lineplot(data=df, x='date', y='emp_count')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/last_week/1.png')
	plt.close()


# Create your views here.
def home(request):
	return render(request, 'recognition/home.html')

@login_required
def dashboard(request):
	if request.user.username == 'admin':
		print("admin")
		return render(request, 'recognition/admin_dashboard.html')
	else:
		print("not admin")
		return render(request, 'recognition/employee_dashboard.html')

@login_required
def add_photos(request):
	if request.user.username != 'admin':
		return redirect('not-authorised')
	if request.method == 'POST':
		form = usernameForm(request.POST)
		data = request.POST.copy()
		username = data.get('username')
		if username_present(username):
			create_dataset(username)
			messages.success(request, f'Dataset Created')
			return redirect('add-photos')
		else:
			messages.warning(request, f'No such username found. Please register employee first.')
			return redirect('dashboard')
	else:
		form = usernameForm()
		return render(request, 'recognition/add_photos.html', {'form': form})

def mark_your_attendance(request):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
	svc_save_path = "face_recognition_data/svc.sav"

	with open(svc_save_path, 'rb') as f:
		svc = pickle.load(f)
	fa = FaceAligner(predictor, desiredFaceWidth=96)
	encoder = LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')

	faces_encodings = np.zeros((1, 128))
	present = dict()
	recognized_user = None
	recognition_start = None

	vs = VideoStream(src=0).start()
	start_time = time.time()

	while (time.time() - start_time) < 10:  # 10-second recognition window
		frame = vs.read()
		frame = imutils.resize(frame, width=800)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray_frame, 0)

		current_face = None
		for face in faces:
			(x, y, w, h) = face_utils.rect_to_bb(face)
			face_aligned = fa.align(frame, gray_frame, face)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

			(pred, prob) = predict(face_aligned, svc)
			
			if pred != [-1]:
				person_name = encoder.inverse_transform(np.ravel([pred]))[0]
				current_face = person_name
				
				if isinstance(prob, np.ndarray):
					prob_value = prob[0]
				else:
					prob_value = prob
				
				cv2.putText(frame, f"{person_name} {prob_value:.2f}", (x+6, y+h-6), 
						   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
				
				if person_name == recognized_user:
					if time.time() - recognition_start >= 2:
						present[person_name] = True
				else:
					recognized_user = person_name
					recognition_start = time.time()

		cv2.imshow("Mark Attendance - In (10 sec)", frame)
		key = cv2.waitKey(50) & 0xFF
		if key == ord("q"):
			break

	vs.stop()
	cv2.destroyAllWindows()
	
	if recognized_user and present.get(recognized_user, False):
		update_attendance_in_db_in(present)
		messages.success(request, f"Welcome {recognized_user}! Attendance marked.")
	else:
		messages.warning(request, "Attendance not recognized.")
	
	return redirect('home')

def mark_your_attendance_out(request):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
	svc_save_path = "face_recognition_data/svc.sav"

	with open(svc_save_path, 'rb') as f:
		svc = pickle.load(f)
	fa = FaceAligner(predictor, desiredFaceWidth=96)
	encoder = LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')

	faces_encodings = np.zeros((1, 128))
	present = dict()
	recognized_user = None
	recognition_start = None

	vs = VideoStream(src=0).start()
	start_time = time.time()

	while (time.time() - start_time) < 10:  # 10-second recognition window
		frame = vs.read()
		frame = imutils.resize(frame, width=800)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray_frame, 0)

		current_face = None
		for face in faces:
			(x, y, w, h) = face_utils.rect_to_bb(face)
			face_aligned = fa.align(frame, gray_frame, face)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

			(pred, prob) = predict(face_aligned, svc)
			
			if pred != [-1]:
				person_name = encoder.inverse_transform(np.ravel([pred]))[0]
				current_face = person_name
				
				if isinstance(prob, np.ndarray):
					prob_value = prob[0]
				else:
					prob_value = prob
				
				cv2.putText(frame, f"{person_name} {prob_value:.2f}", (x+6, y+h-6), 
						   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
				
				if person_name == recognized_user:
					if time.time() - recognition_start >= 2:
						present[person_name] = True
				else:
					recognized_user = person_name
					recognition_start = time.time()

		cv2.imshow("Mark Attendance - Out (10 sec)", frame)
		key = cv2.waitKey(50) & 0xFF
		if key == ord("q"):
			break

	vs.stop()
	cv2.destroyAllWindows()
	
	if recognized_user and present.get(recognized_user, False):
		update_attendance_in_db_out(present)
		messages.success(request, f"Goodbye {recognized_user}! Attendance marked.")
	else:
		messages.warning(request, "Attendance not recognized.")
	
	return redirect('home')

@login_required
def train(request):
	if request.user.username != 'admin':
		return redirect('not-authorised')

	training_dir = 'face_recognition_data/training_dataset'
	count = 0
	for person_name in os.listdir(training_dir):
		curr_directory = os.path.join(training_dir, person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			count += 1

	X = []
	y = []
	i = 0

	for person_name in os.listdir(training_dir):
		print(str(person_name))
		curr_directory = os.path.join(training_dir, person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			print(str(imagefile))
			image = cv2.imread(imagefile)
			try:
				X.append((face_recognition.face_encodings(image)[0]).tolist())
				y.append(person_name)
				i += 1
			except:
				print("removed")
				os.remove(imagefile)

	targets = np.array(y)
	encoder = LabelEncoder()
	encoder.fit(y)
	y = encoder.transform(y)
	X1 = np.array(X)
	print("shape: " + str(X1.shape))
	np.save('face_recognition_data/classes.npy', encoder.classes_)
	svc = SVC(kernel='linear', probability=True)
	svc.fit(X1, y)
	svc_save_path = "face_recognition_data/svc.sav"
	with open(svc_save_path, 'wb') as f:
		pickle.dump(svc, f)

	vizualize_Data(X1, targets)
	
	messages.success(request, f'Training Complete.')

	return render(request, "recognition/train.html")

@login_required
def not_authorised(request):
	return render(request, 'recognition/not_authorised.html')

@login_required
def view_attendance_home(request):
	total_num_of_emp = total_number_employees()
	emp_present_today = employees_present_today()
	this_week_emp_count_vs_date()
	last_week_emp_count_vs_date()
	return render(request, "recognition/view_attendance_home.html", {'total_num_of_emp': total_num_of_emp, 'emp_present_today': emp_present_today})

@login_required
def view_attendance_date(request):
	if request.user.username != 'admin':
		return redirect('not-authorised')
	qs = None
	time_qs = None
	present_qs = None

	if request.method == 'POST':
		form = DateForm(request.POST)
		if form.is_valid():
			date = form.cleaned_data.get('date')
			print("date:" + str(date))
			for user in User.objects.exclude(username='admin'):
				if date >= user.date_joined.date():
					Present.objects.get_or_create(user=user, date=date, defaults={'present': False})
			time_qs = Time.objects.filter(date=date)
			present_qs = Present.objects.filter(date=date)
			if len(time_qs) > 0 or len(present_qs) > 0:
				qs = hours_vs_employee_given_date(present_qs, time_qs)
				return render(request, 'recognition/view_attendance_date.html', {'form': form, 'qs': qs})
			else:
				messages.warning(request, f'No records for selected date.')
				return redirect('view-attendance-date')
	else:
		form = DateForm()
		return render(request, 'recognition/view_attendance_date.html', {'form': form, 'qs': qs})

@login_required
def export_attendance_date_csv(request):
	if request.user.username != 'admin':
		return redirect('not-authorised')
	
	if request.method == 'POST':
		date_str = request.POST.get('date')
		date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
		
		for user in User.objects.exclude(username='admin'):
			if date >= user.date_joined.date():
				Present.objects.get_or_create(user=user, date=date, defaults={'present': False})
		
		present_qs = Present.objects.filter(date=date)
		time_qs = Time.objects.filter(date=date)
		
		qs = hours_vs_employee_given_date(present_qs, time_qs)
		
		response = HttpResponse(content_type='text/csv')
		response['Content-Disposition'] = f'attachment; filename="attendance_{date}.csv"'
		
		writer = csv.writer(response)
		writer.writerow(['Username', 'Date', 'Present', 'Time In', 'Time Out', 'Hours', 'Break Hours'])
		
		for obj in qs:
			writer.writerow([
				obj.user.username,
				obj.date,
				obj.present,
				obj.time_in,
				obj.time_out,
				obj.hours_display,
				obj.break_hours_display
			])
		
		return response
	
	return redirect('view-attendance-date')

@login_required
def view_attendance_employee(request):
	if request.user.username != 'admin':
		return redirect('not-authorised')
	time_qs = None
	present_qs = None
	qs = None

	if request.method == 'POST':
		form = UsernameAndDateForm(request.POST)
		if form.is_valid():
			username = form.cleaned_data.get('username')
			if username_present(username):
				u = User.objects.get(username=username)
				
				time_qs = Time.objects.filter(user=u)
				present_qs = Present.objects.filter(user=u)
				date_from = form.cleaned_data.get('date_from')
				date_to = form.cleaned_data.get('date_to')
				
				if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-attendance-employee')
				else:
					reg_date = u.date_joined.date()
					current_date = max(date_from, reg_date)
					while current_date <= date_to:
						Present.objects.get_or_create(user=u, date=current_date, defaults={'present': False})
						current_date += datetime.timedelta(days=1)
					

					time_qs = time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs = present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					
					if len(time_qs) > 0 or len(present_qs) > 0:
						qs = hours_vs_date_given_employee(present_qs, time_qs, admin=True)
						return render(request, 'recognition/view_attendance_employee.html', {'form': form, 'qs': qs})
					else:
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-attendance-employee')
			else:
				print("invalid username")
				messages.warning(request, f'No such username found.')
				return redirect('view-attendance-employee')
	else:
		form = UsernameAndDateForm()
		return render(request, 'recognition/view_attendance_employee.html', {'form': form, 'qs': qs})

@login_required
def export_attendance_employee_csv(request):
	if request.user.username != 'admin':
		return redirect('not-authorised')
	
	if request.method == 'POST':
		username = request.POST.get('username')
		date_from_str = request.POST.get('date_from')
		date_to_str = request.POST.get('date_to')
		
		date_from = datetime.datetime.strptime(date_from_str, '%Y-%m-%d').date()
		date_to = datetime.datetime.strptime(date_to_str, '%Y-%m-%d').date()
		
		if username_present(username):
			u = User.objects.get(username=username)
			
			reg_date = u.date_joined.date()
			current_date = max(date_from, reg_date)
			while current_date <= date_to:
				Present.objects.get_or_create(user=u, date=current_date, defaults={'present': False})
				current_date += datetime.timedelta(days=1)
			
			present_qs = Present.objects.filter(user=u, date__gte=date_from, date__lte=date_to)
			time_qs = Time.objects.filter(user=u, date__gte=date_from, date__lte=date_to)
			
			qs = hours_vs_date_given_employee(present_qs, time_qs, admin=True)
			
			response = HttpResponse(content_type='text/csv')
			response['Content-Disposition'] = f'attachment; filename="attendance_{username}_{date_from}_to_{date_to}.csv"'
			
			writer = csv.writer(response)
			writer.writerow(['Username', 'Date', 'Present', 'Time In', 'Time Out', 'Hours', 'Break Hours'])
			
			for obj in qs:
				writer.writerow([
					obj.user.username,
					obj.date,
					obj.present,
					obj.time_in,
					obj.time_out,
					obj.hours_display,
					obj.break_hours_display
				])
			
			return response
	
	return redirect('view-attendance-employee')

@login_required
def view_my_attendance_employee_login(request):
	if request.user.username == 'admin':
		return redirect('not-authorised')
	qs = None
	time_qs = None
	present_qs = None
	if request.method == 'POST':
		form = DateForm_2(request.POST)
		if form.is_valid():
			u = request.user
			time_qs = Time.objects.filter(user=u)
			present_qs = Present.objects.filter(user=u)
			date_from = form.cleaned_data.get('date_from')
			date_to = form.cleaned_data.get('date_to')
			if date_to < date_from:
				messages.warning(request, f'Invalid date selection.')
				return redirect('view-my-attendance-employee-login')
			else:
				reg_date = u.date_joined.date()
				current_date = max(date_from, reg_date)
				while current_date <= date_to:
					Present.objects.get_or_create(user=u, date=current_date, defaults={'present': False})
					current_date += datetime.timedelta(days=1)
			

					time_qs = time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs = present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
				
					if len(time_qs) > 0 or len(present_qs) > 0:
						qs = hours_vs_date_given_employee(present_qs, time_qs, admin=False)
						return render(request, 'recognition/view_my_attendance_employee_login.html', {'form': form, 'qs': qs})
					else:
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-my-attendance-employee-login')
	else:
		form = DateForm_2()
		return render(request, 'recognition/view_my_attendance_employee_login.html', {'form': form, 'qs': qs})


@login_required
def export_my_attendance_csv(request):
	if request.user.username == 'admin':
		return redirect('not-authorised')
	
	if request.method == 'POST':
		date_from_str = request.POST.get('date_from')
		date_to_str = request.POST.get('date_to')
		
		date_from = datetime.datetime.strptime(date_from_str, '%Y-%m-%d').date()
		date_to = datetime.datetime.strptime(date_to_str, '%Y-%m-%d').date()
		
		u = request.user
		
		reg_date = u.date_joined.date()
		current_date = max(date_from, reg_date)
		while current_date <= date_to:
			Present.objects.get_or_create(user=u, date=current_date, defaults={'present': False})
			current_date += datetime.timedelta(days=1)
		
		present_qs = Present.objects.filter(user=u, date__gte=date_from, date__lte=date_to)
		time_qs = Time.objects.filter(user=u, date__gte=date_from, date__lte=date_to)
		
		qs = hours_vs_date_given_employee(present_qs, time_qs, admin=False)
		
		response = HttpResponse(content_type='text/csv')
		response['Content-Disposition'] = f'attachment; filename="my_attendance_{date_from}_to_{date_to}.csv"'
		
		writer = csv.writer(response)
		writer.writerow(['Username', 'Date', 'Present', 'Time In', 'Time Out', 'Hours', 'Break Hours'])
		
		for obj in qs:
			writer.writerow([
				obj.user.username,
				obj.date,
				obj.present,
				obj.time_in,
				obj.time_out,
				obj.hours_display,
				obj.break_hours_display
			])
		
		return response
	
	return redirect('view-my-attendance-employee-login')